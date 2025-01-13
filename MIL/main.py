import time
import torch
import wandb
import numpy as np
from copy import deepcopy
import torch.nn as nn
import pandas as pd
from dataloader import *
from model import *
from loss import *
from torch.utils.data import DataLoader, RandomSampler
import argparse, os
from torch.nn.functional import one_hot
from contextlib import suppress
import time
import yaml

from modules.vit_wsi.model_v1 import Model_V1
from timm.utils import AverageMeter,dispatch_clip_grad
from timm.models import  model_parameters
from collections import OrderedDict
from utils import *


def main(args):
    # set seed
    seed_torch(args.seed)
    
    # --->划分数据集, 一类是camelyon16采用n-fold交叉验证，另一类是tct-gc使用划分好的数据集
    if args.datasets.lower() == 'camelyon16':       
            label_path=os.path.join(args.dataset_root,'label.csv')
            p, l = get_patient_label(label_path)
            index = [i for i in range(len(p))]
            random.shuffle(index)
            p = p[index]
            l = l[index]
            
    elif args.datasets.lower() in ['ngc', 'gc', 'fnac', 'gc_mtl']:
        ps, ls = [], []
        label_paths = [args.train_label_path, args.val_label_path, args.test_label_path]
        for label_path in label_paths:
            df = pd.read_csv(label_path)
            p, l = df.iloc[:, 0].values, df.iloc[:, 1].values
            ps.append(p)
            ls.append(l)
        train_p, val_p, test_p = ps
        train_l, val_l, test_l = ls
    if args.cv_fold > 1:
        train_p, train_l, test_p, test_l, val_p, val_l = get_kflod(args.cv_fold, p, l,args.val_ratio)
    else:
        train_p, train_l, test_p, test_l, val_p, val_l = [train_p], [train_l], [test_p], [test_l], [val_p], [val_l]
    
    acs, pre, rec,fs,auc, te_auc, te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec, fs, auc, te_auc, te_fs] # acs: [fold, fold] fold: [task1, task2]

    if not args.no_log:
        print('Dataset: ' + args.datasets)

    # resume
    # if args.auto_resume and not args.no_log:
    #     ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
    #     args.fold_start = ckp['k']
    #     if len(ckp['ckc_metric']) == 6:
    #         acs, pre, rec, fs, auc, te_auc = ckp['ckc_metric']
    #     elif len(ckp['ckc_metric']) == 7:
    #         acs, pre, rec, fs, auc, te_auc, te_fs = ckp['ckc_metric']
    #     else:
    #         acs, pre, rec,fs,auc = ckp['ckc_metric']
    
    for k in range(args.fold_start, args.cv_fold):
        if not args.no_log:
            print('Start %d-fold cross validation: fold %d ' % (args.cv_fold, k))
        ckc_metric = one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l)

    # if args.wandb:
    #     wandb.log({
    #         "cross_val/acc_mean":np.mean(np.array(acs)),
    #         "cross_val/auc_mean":np.mean(np.array(auc)),
    #         "cross_val/f1_mean":np.mean(np.array(fs)),
    #         "cross_val/pre_mean":np.mean(np.array(pre)),
    #         "cross_val/recall_mean":np.mean(np.array(rec)),
    #         "cross_val/acc_std":np.std(np.array(acs)),
    #         "cross_val/auc_std":np.std(np.array(auc)),
    #         "cross_val/f1_std":np.std(np.array(fs)),
    #         "cross_val/pre_std":np.std(np.array(pre)),
    #         "cross_val/recall_std":np.std(np.array(rec)),
    #     })
    # if not args.no_log:
    #     print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
    #     print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
    #     print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
    #     print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
    #     print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))


def one_fold(args,k,ckc_metric,train_p, train_l, test_p, test_l,val_p,val_l):
    # --->initiation
    amp_autocast = torch.cuda.amp.autocast if args.amp else suppress
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric

    # ***--->load data
    if args.datasets.lower() == 'gc_mtl':
        train_set = GcMTLDataset(train_p[k],train_l[k],root=args.dataset_root,num_task=args.num_task, \
                                 num_classes=args.num_classes,persistence=args.persistence, \
                                     keep_same_psize=args.same_psize,is_train=True)
        test_set = GcMTLDataset(test_p[k],test_l[k],root=args.dataset_root,num_task=args.num_task,num_classes=args.num_classes,persistence=args.persistence,keep_same_psize=args.same_psize)
        val_set = GcMTLDataset(val_p[k],val_l[k],root=args.dataset_root,num_task=args.num_task,num_classes=args.num_classes,persistence=args.persistence,keep_same_psize=args.same_psize)
    else:
        train_set = C16Dataset(train_p[k],train_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize,is_train=True)
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        val_set = C16Dataset(val_p[k],val_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
    
    if args.datasets.lower() == 'camelyon16':
        if args.val_ratio == 0.:
            val_set = test_set
            
    if args.imbalance_sampler:
        train_set = ClassBalancedDataset(train_set, oversample_thr=0.22)
    
    if args.fix_loader_random:
        # generated by int(torch.empty((), dtype=torch.int64).random_().item())
        big_seed_list = 7784414403328510413
        generator = torch.Generator()
        generator.manual_seed(big_seed_list)  
        # TODO 增添补齐的collect_fn
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,generator=generator)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=RandomSampler(train_set), num_workers=args.num_workers)
        
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ***--->construct model
    model = MIL_MTL(input_dim=args.input_dim, 
                num_classes=args.num_classes, 
                num_task=args.num_task,
                mil=args.mil_method,
                dropout=args.dropout, 
                act=args.act).to(device)

    # ***--->construct criterion
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'ce':
        criterion = nn.CrossEntropyLoss()
    elif args.loss == 'softbce':
        criterion = MySoftBCELoss(neg_weight=args.neg_weight)
    elif args.loss == 'ranking':
        criterion = RankingAndSoftBCELoss(neg_weight=args.neg_weight, neg_margin=args.neg_margin)
    elif args.loss == 'aploss':
        criterion = APLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss(alpha=torch.tensor([0.5, 1, 1, 1, 1]))

    # optimizer
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.lr_sche == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, 0) if not args.lr_supi else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch*len(train_loader), 0)
    elif args.lr_sche == 'step':
        assert not args.lr_supi
        # follow the DTFD-MIL
        # ref:https://github.com/hrzhang1123/DTFD-MIL
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,args.num_epoch / 2, 0.2)
    elif args.lr_sche == 'const':
        scheduler = None

    if args.early_stopping:
        early_stopping = EarlyStopping(patience=30 if args.datasets=='camelyon16' else 20, stop_epoch=args.max_epoch if args.datasets=='camelyon16' else 70,save_best_model_stage=np.ceil(args.save_best_model_stage * args.num_epoch))
    else:
        early_stopping = None

    opt_ac, opt_pre, opt_re, opt_fs, opt_auc, opt_epoch = 0, 0, 0, 0, 0, 0
    epoch_start = 0

    # resume
    # if args.auto_resume and not args.no_log:
    #     ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
    #     epoch_start = ckp['epoch']
    #     model.load_state_dict(ckp['model'])
    #     optimizer.load_state_dict(ckp['optimizer'])
    #     scheduler.load_state_dict(ckp['lr_sche'])
    #     early_stopping.load_state_dict(ckp['early_stop'])
    #     opt_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch = ckp['val_best_metric']
    #     np.random.set_state(ckp['random']['np'])
    #     torch.random.set_rng_state(ckp['random']['torch'])
    #     random.setstate(ckp['random']['py'])
    #     if args.fix_loader_random:
    #         train_loader.sampler.generator.set_state(ckp['random']['loader'])
    #     args.auto_resume = False

    train_time_meter = AverageMeter()

    # 如果只用来评估测试集的性能
    if args.eval_only:
        ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
        model.load_state_dict(ckp['model'])
        val_loop(args,model,test_loader,device,criterion,early_stopping,epoch=0,test_mode=True,\
            data_split='test')
        return
    
    # 训练epoch
    for epoch in range(epoch_start, args.num_epoch):
        train_loss,start,end = train_loop(args,model,train_loader,optimizer,device,amp_autocast,criterion,scheduler,k,epoch)
        train_time_meter.update(end-start)
        
        # train_set acc
        print('Info: Evaluation for train set')
        val_loop(args,model,train_loader,device,criterion, early_stopping,epoch,test_mode=True,data_split='train')
        
        print('Info: Evaluation for val set')
        stop, aucs, acc, recs, precs, f1s, test_loss = val_loop(args,model,val_loader,device,criterion,\
            early_stopping,epoch,data_split='val')

        if not args.no_log:
            print('\r Epoch [%d/%d] train loss: %.1E, test loss: %.1E, time: %.3f(%.3f)' % 
        (epoch+1, args.num_epoch, train_loss, test_loss, train_time_meter.val,train_time_meter.avg))
           
        auc_mean, acc_mean, rec_mean, prec_mean, f1_mean = np.mean(aucs), acc, np.mean(recs), np.mean(precs), np.mean(f1s)
        if rec_mean > opt_re and epoch >= args.save_best_model_stage*args.num_epoch:
            opt_auc, opt_ac, opt_rec, opt_prec, opt_f1, opt_epoch = auc_mean, acc_mean, rec_mean, prec_mean, f1_mean, epoch
            if not os.path.exists(args.model_path):
                os.mkdir(args.model_path)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                }
                torch.save(best_pt, os.path.join(args.model_path, 'fold_{fold}_model_best_recall.pt'.format(fold=k)))
        # if args.wandb:
        #     rowd = OrderedDict([
        #         ("val_best_acc",opt_ac),
        #         ("val_best_precesion",opt_pre),
        #         ("val_best_recall",opt_re),
        #         ("val_best_fscore",opt_fs),
        #         ("val_best_auc",opt_auc),
        #         ("val_best_epoch",opt_epoch),
        #     ])

        #     rowd = OrderedDict([ (str(k)+'-fold/'+_k,_v) for _k, _v in rowd.items()])
        #     wandb.log(rowd)
        
        # save checkpoint
        random_state = {
            'np': np.random.get_state(),
            'torch': torch.random.get_rng_state(),
            'py': random.getstate(),
            'loader': train_loader.sampler.generator.get_state() if args.fix_loader_random else '',
        }
        ckp = {
            'model': model.state_dict(),
            'lr_sche': scheduler.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch+1,
            'k': k,
            'early_stop': early_stopping.state_dict(),
            'random': random_state,
            'ckc_metric': [acs,pre,rec,fs,auc,te_auc,te_fs],
            'val_best_metric': [opt_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch],
            'wandb_id': wandb.run.id if args.wandb else '',
        }
        if not args.no_log:
            torch.save(ckp, os.path.join(args.model_path, 'ckp.pt'))

        if stop:
            break

    # test
    if not args.no_log:
        best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_recall.pt'.format(fold=k)))
        info = model.load_state_dict(best_std['model'])
        print(info)
        
    print('Info: Evaluation for test set')
    aucs, acc, recs, precs, f1s, test_loss = val_loop(args,model,test_loader,device,criterion,early_stopping,\
        epoch,test_mode=True,data_split='test')
    res = OrderedDict([
            ("test_auc_mean", np.mean(aucs)),
            ("test_acc_mean", acc),
            ("test_recall_mean", np.mean(recs)),
            ("test_precision_mean", np.mean(precs)),
            ("test_fscore_mean", np.mean(f1s)),
            ("test_loss",test_loss.cpu()),
        ])
    df = pd.DataFrame(res, index=[1])
    df.to_excel(os.path.join(args.model_path, 'evaluation.xlsx'), index=False)
    
    if not args.no_log:
        print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' \
            % (opt_ac,opt_auc,opt_pre,opt_re,opt_fs))
    auc_fold, acc_fold, rec_fold, prec_fold, f1_fold = [], [], [], [], []
    auc_fold.append(np.mean(aucs))
    acc_fold.append(acc)
    rec_fold.append(np.mean(recs))
    prec_fold.append(np.mean(precs))
    f1_fold.append(f1s)
    return [auc_fold,acc_fold,rec_fold,prec_fold,f1_fold]

def train_loop(args,model,loader,optimizer,device,amp_autocast,criterion,scheduler,k,epoch):
    start = time.time()
    loss_cls_meter = AverageMeter()
    train_loss_log = 0.
    model.train()
    sum_num_classes = sum(args.num_classes)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_param = sum(p.numel() for p in model.parameters())
    if not args.no_log:
        print('Number of total parameters: {}, tunable parameters: {}'.format(num_total_param, n_parameters))
    
    for i, data in enumerate(loader):
        optimizer.zero_grad()

        bag, label, task_id = data[0].to(device), data[1].to(device), data[2].to(device)  # [b,n,1024]
        batch_size=bag.size(0)
            
        with amp_autocast():
            if args.patch_shuffle:
                bag = patch_shuffle(bag,args.shuffle_group)
            elif args.group_shuffle:
                bag = group_shuffle(bag,args.shuffle_group)

            # TODO 一阶段方法传入bag, task_id , 两阶段方法传入bag, mask 
            train_logits = model(bag, task_id) # [B, sum_num_classes]
            # if args.loss != 'ce':
            #     train_logits = torch.sigmoid(train_logits)    
            
            if args.loss in ['ce', 'focal']:
                logit_loss = criterion(train_logits.view(batch_size,-1),label)
            elif args.loss in ['bce', 'softbce', 'ranking']:
                logit_loss = criterion(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1),num_classes=sum_num_classes).squeeze(1).float())
            elif args.loss == 'aploss':
                logit_loss = criterion.apply(train_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1),num_classes=sum_num_classes).squeeze(1).float())
            assert not torch.isnan(logit_loss)

        train_loss = logit_loss

        train_loss = train_loss / args.accumulation_steps
        if args.clip_grad > 0.:
            dispatch_clip_grad(
                model_parameters(model),
                value=args.clip_grad, mode='norm')

        if (i+1) % args.accumulation_steps == 0:
            train_loss.backward()
            optimizer.step()
            if args.lr_supi and scheduler is not None:
                scheduler.step()
            
        loss_cls_meter.update(logit_loss,1)

        if i % args.log_iter == 0 or i == len(loader)-1:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            rowd = OrderedDict([
                ('train_loss', train_loss),
                ('lr',lr),
            ])
            if not args.no_log:
                print('[{}/{}] logit_loss:{}'.format(i,len(loader)-1,loss_cls_meter.avg))
            rowd = OrderedDict([ (_k,_v) for _k, _v in rowd.items()])
            if args.wandb:
                wandb.log(rowd)

        train_loss_log = train_loss_log + train_loss.item()

    end = time.time()
    
    train_loss_log = train_loss_log/len(loader)
    if not args.lr_supi and scheduler is not None:
        scheduler.step()
    
    return train_loss_log,start,end

def val_loop(args,model,loader,device,criterion,early_stopping,epoch,test_mode=False,data_split='val'):
    """
    返回:
        stop (bool): 是否earlystop
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
        loss_cls_meter (float): 损失
    """
    model.eval()
    loss_cls_meter = AverageMeter()
    all_class_labels = []
    for _class_labels in args.class_labels:
        all_class_labels.extend(_class_labels)
    sum_num_classes=len(all_class_labels)
    
    bag_logits, bag_labels, wsi_names = [], [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            bag, label, task_id = data[0].to(device), data[1].to(device), data[2] # b*n*1024
            wsi_name = [os.path.basename(_) for _ in data[3]]
            
            test_logits = model(bag, task_id)
            batch_size = bag.size(0)
            bag_labels.extend(data[1])
            wsi_names.extend(wsi_name)
            
            if args.loss in ['ce']:
                test_loss = criterion(test_logits.view(batch_size,-1),label)    
                if len(all_class_labels) == 2:
                    bag_logits.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().numpy())
                else:
                    bag_logits.extend(torch.softmax(test_logits, dim=-1).cpu().numpy())
            # TODO have not updated            
            elif args.loss in ['bce', 'softbce', 'ranking', 'aploss', 'focal']:
                if args.loss in ['focal']:
                    test_loss = criterion(test_logits.view(batch_size,-1),label)
                elif args.loss in ['bce', 'softbce', 'ranking']:
                    test_loss = criterion(test_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1),num_classes=sum_num_classes).squeeze(1).float())
                elif args.loss == 'aploss':
                    test_loss = criterion.apply(test_logits.view(batch_size,-1),one_hot(label.view(batch_size,-1),num_classes=sum_num_classes).squeeze(1).float())
                
                if all_class_labels == 2:
                    bag_logits.extend(torch.sigmoid(test_logits)[:,1].cpu().numpy())
                else:
                    bag_logits.extend(torch.sigmoid(test_logits).cpu().numpy())
            loss_cls_meter.update(test_loss,1)
    
    aucs, acc, recs, precs, f1s = multi_class_scores_mtl(bag_labels, bag_logits, all_class_labels, wsi_names, args.threshold, args.eval_only)
    
    # cancer任务的指标
    aucs_task1, recs_task1, precs_task1, f1s_task1 = aucs[:5], recs[:5], precs[:5], f1s[:5]
    rowd = OrderedDict([(f"auc",np.mean(aucs_task1)),  (f"recall",np.mean(recs_task1)),\
                        (f"precision",np.mean(precs_task1)),(f"fscore",np.mean(f1s_task1)),])
    rowd = OrderedDict([ (f'cancer-task/{data_split}_'+_k,_v) for _k, _v in rowd.items()]+[('epoch',epoch)])
    if args.wandb:
        wandb.log(rowd)
    print(rowd)
    # microbial任务的指标
    if len(aucs) > 4:
        aucs_task2, recs_task2, precs_task2, f1s_task2 = aucs[5:], recs[5:], precs[5:], f1s[5:]
        rowd = OrderedDict([(f"auc",np.mean(aucs_task2)),  (f"recall",np.mean(recs_task2)),\
                            (f"precision",np.mean(precs_task2)),(f"fscore",np.mean(f1s_task2)),])
        rowd = OrderedDict([ (f'microbial-task/{data_split}_'+_k,_v) for _k, _v in rowd.items()]+[('epoch',epoch)])
        if args.wandb:
            wandb.log(rowd)
        
        print(rowd)
    # 两个任务的平均指标
    rowd = OrderedDict([(f"auc",np.mean(aucs)),  (f"recall",np.mean(recs)),\
                        (f"precision",np.mean(precs)),(f"fscore",np.mean(f1s)), (f"accuracy", acc)])
    rowd = OrderedDict([ (f'{data_split}_'+_k,_v) for _k, _v in rowd.items()]+[('epoch',epoch)])
    if args.wandb:
        wandb.log(rowd)
    print(rowd)
    if args.loss in ['ce']:
        loss_cls_meter = loss_cls_meter.avg
    else:
        loss_cls_meter = loss_cls_meter.sum
    if test_mode:
        return aucs, acc, recs, precs, f1s, loss_cls_meter
    else:
        # val mode detect early stopping
        # early stop
        if early_stopping is not None:
            early_stopping(epoch,np.mean(recs),model)
            stop = early_stopping.early_stop
        else:
            stop = False
        return stop, aucs, acc, recs, precs, f1s, loss_cls_meter

if __name__ == '__main__':
    
    # args = argparse.ArgumentParser()
    # args.in_dims = 256
    # args.num_classes = [1,5,3]
    # args.depth = 4
    # args.num_heads = 4
    # args.proj_drop = 0.02
    # args.attn_drop = 0.02
    # args.drop_path = 0.01

    # model = Model_V1(
    #     in_dims=args.in_dims,
    #     num_classes=args.num_classes,
    #     depth=args.depth,
    #     num_heads=args.num_heads,
    #     proj_drop=args.proj_drop,
    #     attn_drop=args.attn_drop,
    #     drop_path=args.drop_path,
    # )

    # size = (5, 10, 100, 256) # [B, N, M, C]
    # mask = torch.ones(size[0], size[1], size[2])
    # a = torch.randn(size)
    # output = model(a, mask)
    # print(output.shape)
    
    # assert 1==0
    
    parser = argparse.ArgumentParser(description='MIL Training Script')

    # Dataset 
    parser.add_argument('--datasets', default='gc_mtl', type=str, help='[camelyon16, tcga, ngc, gc, fnac, gc_mtl, gc_fine]')
    parser.add_argument('--dataset_root', default='/home1/wsi/gc-all-features/frozen/gigapath-longnet', type=str, help='Dataset root path')
    parser.add_argument('--label_path', default='../datatools/gc-2000/onetask_labels', type=str, help='label of train dataset')
    parser.add_argument('--imbalance_sampler', default=0, type=float, help='if use imbalance_sampler')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--val_ratio', default=0., type=float, help='Val-set ratio')
    parser.add_argument('--fold_start', default=0, type=int, help='Start validation fold [0]')
    parser.add_argument('--cv_fold', default=1, type=int, help='Number of cross validation fold [3]')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')
    parser.add_argument('--train_val', default=0, type=int, help='use train and val set to train the model')

    # Train
    parser.add_argument('--auto_resume', action='store_true', help='Resume from the auto-saved checkpoint')
    parser.add_argument('--num_epoch', default=200, type=int, help='Number of total training epochs [200]')
    parser.add_argument('--early_stopping', action='store_false', help='Early stopping')
    parser.add_argument('--max_epoch', default=130, type=int, help='Number of max training epochs in the earlystopping [130]')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of batch size')
    
    # Loss
    parser.add_argument('--loss', default='focal', type=str, help='Classification Loss [ce, bce, softbce, ranking, aploss, focal]')
    parser.add_argument('--neg_weight', default=0.0, type=float, help='Weight for positive sample in SoftBCE')
    parser.add_argument('--neg_margin', default=0, type=float, help='if use neg_margin in ranking loss')
    parser.add_argument('--opt', default='adam', type=str, help='Optimizer [adam, adamw]')
    parser.add_argument('--save_best_model_stage', default=0., type=float, help='See DTFD')
    parser.add_argument('--seed', default=2024, type=int, help='random number [2021]' )
    parser.add_argument('--lr', default=2e-3, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--lr_sche', default='cosine', type=str, help='Deacy of learning rate [cosine, step, const]')
    parser.add_argument('--lr_supi', action='store_true', help='LR scheduler update per iter')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--accumulation_steps', default=1, type=int, help='Gradient accumulate')
    parser.add_argument('--clip_grad', default=.0, type=float, help='Gradient clip')

    # Model
    # mil meathod
    parser.add_argument('--mil_method', default='abmil', type=str, help='Model name [abmil, transmil, dsmil, clam, linear]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')
    parser.add_argument('--input_dim', default=768, type=int, help='The dimention of patch feature')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # Misc
    parser.add_argument('--project', default='mtl-524', type=str, help='Project name of exp')
    parser.add_argument('--title', default='test', type=str, help='Title of exp')
    parser.add_argument('--log_iter', default=100, type=int, help='Log Frequency')
    parser.add_argument('--amp', action='store_true', help='Automatic Mixed Precision Training')
    parser.add_argument('--wandb', action='store_true', help='Weight&Bias')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--nonilm', type=float, default=0, help='no nilm')
    parser.add_argument('--model_path', type=str, default='./output-model', help='Output path')
    parser.add_argument('--task_config', type=str, default='./configs/oh_5.yaml', help='Task config path')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    parser.add_argument('--threshold', default=0.3, type=float, help='threshold for evaluation')
    
    
    args = parser.parse_args()
    
    with open(args.task_config, 'r') as f:
        task_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        args.num_task = task_config['num_task']
        args.num_classes = task_config['num_classes']
        args.class_labels = task_config['class_labels']
        # args.label_path = task_config['label_path']
        f.close()
    # args.num_task = 3
    # args.num_classes = [2,2,4]
    # args.class_labels = [['NILM', 'POS'], ['NILM', 'AGC'], ['NILM', 'T', 'M', 'BV']]
    
    if args.train_val:
        args.train_label_path = os.path.join(args.label_path, 'train_val.csv')
        args.val_label_path = os.path.join(args.label_path, 'test_label.csv')
        args.test_label_path = os.path.join(args.label_path, 'test_label.csv')
    else:
        args.train_label_path = os.path.join(args.label_path, 'train_label.csv')
        args.val_label_path = os.path.join(args.label_path, 'val_label.csv')
        args.test_label_path = os.path.join(args.label_path, 'test_label.csv')
    
    if not os.path.exists(os.path.join(args.model_path,args.project)):
        os.mkdir(os.path.join(args.model_path,args.project))
    # args.model_path = os.path.join(args.model_path,args.project,args.title)
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    args.fix_loader_random = True
    args.fix_train_random = True
    
    # wandb注册
    if args.wandb:
        wandb.login()
        if args.auto_resume:
            ckp = torch.load(os.path.join(args.model_path,'ckp.pt'))
            wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(args.model_path),id=ckp['wandb_id'],resume='must')
        else:
            wandb.init(project=args.project, name=args.title,config=args,dir=os.path.join(args.model_path))
        
    print(args)
    localtime = time.asctime(time.localtime(time.time()) )
    print(localtime)
    main(args=args)
