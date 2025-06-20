import argparse
import time
import os
import torch
import torch.nn.functional as F
from datasets import C16Dataset
from models import SimCLR_MIL
from utils import nt_xent, focal_nt_xent, GHLoss, lr_scheduler, lr_scheduler_warm, eval, AverageMeter, logger, disjoint_summary
from tct_utils import PatchFeatureAugmenter, TwoViewAugDataset_index
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import numpy as np
import pandas as pd
import wandb
from BCL.memoboosted_cifar100 import Memoboosed_Dataset
from SDCLR.sdclr import SDCLR, Mask

def ddp_setup(rank: int, world_size: int):
    """
    Args:
       rank: Unique identifier of each process
      world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
   
def ssl_training(rank, world_size, args):
    ddp_setup(rank, world_size)

    # gpus
    # gpus = list(map(lambda x: torch.device('cuda', x), [int(e) for e in args.gpus.strip().split(",")]))
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True
    if args.wandb:
        wandb.init(project='Geometric Harmonization', config=args, name=args.wandb_name, settings=wandb.Settings(start_method="fork"))

    if os.path.exists(args.log_folder) is not True:
        os.system("mkdir -p {}".format(args.log_folder))

    log = logger(path=args.log_folder, log_name="log.txt")

    # dataset
    # if args.dataset == 'tct':
    #     train_dataset = ImbalanceCIFAR100("dataset/", imb_type=args.imb_type, imb_factor=args.imb_factor,
    #                                     rand_number=0, train=True, download=True)
    #     num_classes = 100

    ################# custome dataset #################
    id2label = {
    0: 'nilm',
    1: 'ascus',
    2: 'asch',
    3: 'lsil',
    4: 'hsil',
    5: 'agc',
    6: 't',
    7: 'm',
    8: 'bv',}
    label2id = {v: k for k, v in id2label.items()}
    if args.dataset == 'gc_10k':
        args.project = 'gc_10k/warmup-rankloss'
        args.train_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-train.csv'
        args.test_label_path = '/data/wsi/TCTGC10k-labels/6_labels/TCTGC10k-v15-test.csv'
        args.train_cluster_path = f'../datatools/TCTGC10k/cluster/kmeans_{args.kmeans_k}.csv'
        args.dataset_root = '/data/wsi/TCTGC50k-features/gigapath-coarse'
        # args.dataset_root = '/data/wsi/TCTGC10k-features/gigapath-1000'
        num_classes = 6
        args.class_labels = ['nilm', 'ascus', 'asch', 'lsil', 'hsil', 'agc']
        df_train = pd.read_csv(args.train_label_path)
        df_cluster = pd.read_csv(args.train_cluster_path)
        # 合并到训练集 DataFrame，确保 wsi_name 对齐
        df_train = df_train.merge(
            df_cluster[['wsi_name', 'cluster_label']], 
            on='wsi_name', 
            how='left'  # 保留所有训练集样本，缺失的聚类标签设为 NaN
        )
        train_wsi_names = df_train['wsi_name'].values
        train_wsi_labels = df_train['wsi_label'].map(label2id).values
        df_test = pd.read_csv(args.test_label_path)
        test_wsi_names = df_test['wsi_name'].values
        test_wsi_labels = df_test['wsi_label'].map(label2id).values
        train_cluster_labels = df_train['cluster_label'].apply(
            lambda x: [int(i) for i in x.split()]).values
        train_cluster_labels = [torch.tensor(labels).to('cuda') for labels in train_cluster_labels]
        train_dataset = C16Dataset(train_wsi_names, train_wsi_labels, root=args.dataset_root, num_classes=num_classes, cluster_labels=train_cluster_labels)
        
        # val_dataset = C16Dataset(test_wsi_names,test_wsi_labels,root=args.dataset_root)
        # test_dataset = C16Dataset(test_wsi_names,test_wsi_labels,root=args.dataset_root)
    ################# end #################
    
    if args.bcl: # unused
        train_aug_dataset = Memoboosed_Dataset(train_dataset, args)
    else:
        train_aug = PatchFeatureAugmenter()
        train_aug_dataset = TwoViewAugDataset_index(train_dataset, train_aug)
    sampler = DistributedSampler(train_aug_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_aug_dataset, batch_size=args.batch_size,
                                               num_workers=0, shuffle=False,
                                               sampler=sampler)
    class_stat = np.array(train_dataset.get_cls_num_list())
    dataset_total_num = np.sum(class_stat)

    eval_train_aug, eval_test_aug = PatchFeatureAugmenter(), PatchFeatureAugmenter(augment_type='none')

    if rank == 0:
        eval_batch_size = 256
        eval_train_dataset = C16Dataset(train_wsi_names, train_wsi_labels,root=args.dataset_root, transform=eval_train_aug, cluster_labels=train_cluster_labels)
        eval_test_dataset = C16Dataset(test_wsi_names, test_wsi_labels,root=args.dataset_root, transform=eval_test_aug, cluster_labels=train_cluster_labels)
        eval_train_loader = torch.utils.data.DataLoader(eval_train_dataset, batch_size=eval_batch_size,
                                                num_workers=0, shuffle=False, pin_memory=False)
        eval_test_loader = torch.utils.data.DataLoader(eval_test_dataset, batch_size=eval_batch_size,
                                                num_workers=0, shuffle=False, pin_memory=False)

    # model and loss
    if args.method == "sdclr" or args.method == "sdclr_GH":
        model = SDCLR(num_class=num_classes, network=args.network)
    else:
        # model = SimCLR(num_class=num_classes, network=args.network).to(gpus[0])
        model = SimCLR_MIL(num_class=num_classes, mil=args.network)

    model = model.to(device)
    model = DDP(model, device_ids=[rank])
    
    if args.method == "simclr_GH" or args.method == "sdclr_GH" or args.method == "focal_GH":
        GH_loss = GHLoss(0.05, args.sinkhorn_iter)
        
    if args.target_dim <= 128:
        target_weight = np.load('target/{}_128_target.npy'.format(args.target_dim))
    else:
        target_weight = np.load('target/{}_128_approximate_target.npy'.format(args.target_dim))
    target_weight = torch.tensor(target_weight).float().cuda().detach()
    logit_momentum = torch.empty((dataset_total_num, args.target_dim)).cuda()
    momentum_loss = torch.empty((dataset_total_num, 1)).cuda()

    # optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # training
    loss_meter = AverageMeter("loss")
    loss_cl_meter = AverageMeter("loss_cl")
    loss_gh_meter = AverageMeter("loss_gh")
    time_meter = AverageMeter("time")
    log_interval, ckpt_interval, beta = 10, 200, 0.9
    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        if args.warm_up > 0:
            lr_scheduler_warm(optimizer, epoch, args)
            if epoch >= args.warm_up:
                beta = args.beta
        else:
            lr_scheduler(optimizer, epoch, args)
        
        if rank == 0:
            log.info("current lr is {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        model.train()
        loss_meter.reset(), loss_cl_meter.reset(), loss_gh_meter.reset(), time_meter.reset()

        if args.method == "sdclr" or args.method == "sdclr_GH":
            pruneMask = Mask(model)
            pruneMask.magnitudePruning(args.prune_percent - args.random_prune_percent, args.random_prune_percent)

        t0 = time.time()
        for ii, (im_x, im_y, _, index) in enumerate(train_loader):
            
            optimizer.zero_grad()
            batch_size = im_x.shape[0]
            if args.method != "sdclr" and args.method != "sdclr_GH":
                # outs = model(im_x.to(device), im_y.to(device))
                outs = model(im_x, im_y)

            if args.method == "simclr" or (args.method == "simclr_GH" and epoch + 1 <= args.warm_up):

                loss_cl = nt_xent(outs['z1'], t=0.2, features2 = outs['z2'], average=False)
                loss = loss_cl.mean()
                loss_cl_meter.update(loss, batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss, "ep": epoch})

            elif args.method == "simclr_GH" and epoch + 1 > args.warm_up:
                loss_cl = nt_xent(outs['z1'], t=0.2, features2 = outs['z2'], average=False)
                t1 = F.normalize(outs['z1'].mm(target_weight.t()), dim=1)
                t2 = F.normalize(outs['z2'].mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1, t2, pi) 
                if epoch + 1 <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch + 1 - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                loss = loss_cl.mean() + w * loss_gh
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss_cl, "loss_gh": loss_gh, "ep": epoch})

            if args.method == "focal" or (args.method == "focal_GH" and epoch + 1 <= args.warm_up):
                loss_cl = focal_nt_xent(outs['z1'], t=0.2, features2 = outs['z2'], average=False)
                loss = loss_cl.mean()
                loss_cl_meter.update(loss, batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss, "ep": epoch})

            elif args.method == "focal_GH" and epoch + 1 > args.warm_up:
                loss_cl = focal_nt_xent(outs['z1'], t=0.2, features2 = outs['z2'], average=False)
                t1 = F.normalize(outs['z1'].mm(target_weight.t()), dim=1)
                t2 = F.normalize(outs['z2'].mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1, t2, pi) 
                if epoch + 1 <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch + 1 - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                loss = loss_cl.mean() + w * loss_gh
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss_cl, "loss_gh": loss_gh, "ep": epoch})

            elif args.method == "sdclr" or (args.method == "sdclr_GH" and epoch + 1 <= args.warm_up):
                with torch.no_grad():
                    model.backbone.set_prune_flag(True)
                    # features_2_noGrad = model(im_y.to(device)).detach()
                    features_2_noGrad = model(im_y).detach()
                model.backbone.set_prune_flag(False)
                # features_1 = model(im_x.to(device))
                features_1 = model(im_x)
                loss_cl = nt_xent(features_1, t=0.2, features2 = features_2_noGrad, average=False)
                loss_cl.mean().backward()
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss_cl.mean(), "ep": epoch})

                features_1_no_grad = features_1.detach()
                model.backbone.set_prune_flag(True)
                features_2 = model(im_y.to(device))
                features_2 = model(im_y)
                loss = nt_xent(features_1_no_grad, t=0.2, features2 = features_2, average=False).mean()

            elif args.method == "sdclr_GH" and epoch + 1 > args.warm_up:
                if epoch + 1 <= args.warm_up + args.GH_warm_up:
                    w = 1 * (epoch + 1 - args.warm_up) / args.GH_warm_up
                else:
                    w = 1
                with torch.no_grad():
                    model.backbone.set_prune_flag(True)
                    # features_2_noGrad = model(im_y.to(device)).detach()
                    features_2_noGrad = model(im_y).detach()
                model.backbone.set_prune_flag(False)
                # features_1 = model(im_x.to(device))
                features_1 = model(im_x)
                loss_cl = nt_xent(features_1, t=0.2, features2 = features_2_noGrad, average=False)
                t1 = F.normalize(features_1.mm(target_weight.t()), dim=1)
                t2 = F.normalize(features_2_noGrad.mm(target_weight.t()), dim=1)
                loss_gh = GH_loss(t1, t2, pi) 
                loss = loss_cl.mean() + w * loss_gh
                loss.backward()
                loss_cl_meter.update(loss_cl.mean(), batch_size)
                loss_gh_meter.update(loss_gh, batch_size)
                if args.wandb:
                    wandb.log({"loss_cl": loss_cl.mean(), "loss_gh": loss_gh, "ep": epoch})

                features_1_no_grad = features_1.detach()
                model.backbone.set_prune_flag(True)
                # features_2 = model(im_y.to(device))
                features_2 = model(im_y)
                loss_cl_ = nt_xent(features_1_no_grad, t=0.2, features2 = features_2, average=False)
                t1 = F.normalize(features_1_no_grad.mm(target_weight.t()), dim=1)
                t2 = F.normalize(features_2.mm(target_weight.t()), dim=1)
                loss_gh_ = GH_loss(t1, t2, pi) 
      
                loss = loss_cl_.mean() + w * loss_gh_

          
            if args.method == "sdclr" or args.method == "sdclr_GH": 
                feat = features_1.detach().clone()
            else:
                feat = outs['z1'].detach().clone()

            logit = F.normalize(feat.mm(target_weight.t()), dim=1)
            for count in range(batch_size):
                if epoch == 0:
                    logit_momentum[index[count]] += logit[count].clone()
                    if args.bcl:
                        momentum_loss[index[count]] = loss_cl[count].clone().detach()
                else:
                    logit_momentum[index[count]] = beta * logit_momentum[index[count]] \
                                                                + (1.0 - beta) * logit[count].clone()
                    if args.bcl:
                        momentum_loss[index[count]] = (1.0 - args.momentum_loss_beta) * loss_cl[count].clone().detach() + args.momentum_loss_beta * momentum_loss[index[count]]

            loss_meter.update(loss, batch_size)
            if args.wandb:
                wandb.log({"loss": loss, "ep": epoch})

            loss.backward()
            optimizer.step()
            time_meter.update(time.time() - t0)
            
            if rank == 0 and ii % log_interval == 0:
                if args.method == "simclr" or args.method == "sdclr" or args.method == "focal":
                    log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                        f"{loss_meter}\t{time_meter}")
                elif args.method == "simclr_UA":
                    log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                                f"{loss_meter}\t{loss_cl_meter}\t{loss_gh_meter}\t{time_meter}")
                elif args.method == "simclr_GH" or args.method == "sdclr_GH" or args.method == "focal_GH":
                    if epoch<=args.warm_up:
                        log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                            f"{loss_meter}\t{time_meter}")
                    else:
                        log.info(f"Epoch {epoch}/{args.epochs}\tIt {ii}/{len(train_loader)}\t" +
                            f"{loss_meter}\t{loss_cl_meter}\t{loss_gh_meter}\t{time_meter}")
            t0 = time.time()
                     
            if args.method == "simclr_GH" or args.method == "sdclr_GH" or args.method == "focal_GH":
                if (epoch + 1) % 20 == 0 or epoch + 1 == args.warm_up:
                    if epoch + 1 >= args.warm_up:
                        _, target_preds = torch.max(logit_momentum, 1) 
                        idxsPerClass = [np.where(np.array(target_preds.detach().cpu()) == idx)[0] for idx in range(args.target_dim)]
                        idxsNumPerClass = np.array([len(idxs) for idxs in idxsPerClass])
                        pi = torch.tensor(idxsNumPerClass/np.sum(idxsNumPerClass)).cuda().detach()    

        if args.bcl:
            train_aug_dataset.update_momentum_weight(momentum_loss, epoch)

        if rank == 0 and (epoch + 1) % 10 == 0:
            acc, perClassAcc = eval(eval_train_loader, eval_test_loader, model.module, 500, args=args)
            _, majorAccList, moderateAccList, minorAccList, _, _ = disjoint_summary('Eval', acc[1], perClassAcc, class_stat, dataset='cifar100', returnValue=True)
            log.info(f'Accuracy {acc}, Accuracy major {np.mean(majorAccList)}, Accuracy medium {np.mean(moderateAccList)}, Accuracy minor {np.mean(minorAccList)}')
            if args.wandb:
                wandb.log({"acc": acc[1], "acc_major": np.mean(majorAccList), "acc_medium": np.mean(moderateAccList), "acc_minor": np.mean(minorAccList), "ep": epoch})

        # save model for different epochs
        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            ckpt_file = os.path.join(args.log_folder, 'ssl_{}_{}.pth'.format(args.network, epoch + 1))
            torch.save(model.module.state_dict(), ckpt_file)
            log.info(f'Saved to {ckpt_file}')
            
    destroy_process_group()



if __name__ == "__main__":
            
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument("--mode", type=str, default="training", help="training | test")
    parser.add_argument("--gpus", type=str, default="0", help="gpu id sequence split by comma")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')
    parser.add_argument("--network", type=str, default="abmil")
    parser.add_argument("--method", type=str, default="simclr")
    parser.add_argument('--lr', type=float, default=0.5, help='learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='minimum learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument("--epochs", type=int, default=1000, help="training epochs")
    parser.add_argument("--log_folder", type=str, default="./", help="log the epoch-test performance")
    parser.add_argument("--checkpoint_path", type=str, default="", help="checkpoint path")
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument("--wandb_name", type=str, default="")

    parser.add_argument("--warm_up", type=int, default=0, help="warm up epoch for target initialization")
    parser.add_argument('--lr_min_stage1', type=float, default=0.3)
    parser.add_argument('--lr_max_stage2', type=float, default=0.3)
    parser.add_argument("--GH_warm_up", type=int, default=10, help="warm up epoch for target initialization")
    parser.add_argument("--target_dim", type=int, default=20, help="target dimension")
    parser.add_argument('--sinkhorn_iter', default=300, type=int, help='sinkhorn iterations')
    parser.add_argument('--beta', default=0.999, type=float, help='beta')

    # BCL
    parser.add_argument('--bcl', action='store_true', help='boosted contrastive learning')
    parser.add_argument('--momentum_loss_beta', type=float, default=0.97)
    parser.add_argument('--rand_k', type=int, default=1, help='k in randaugment')
    parser.add_argument('--rand_strength', type=int, default=30, help='maximum strength in randaugment(0-30)')
    # SDCLR
    parser.add_argument('--prune_percent', type=float, default=0.9, help="whole prune percentage")
    parser.add_argument('--random_prune_percent', type=float, default=0, help="random prune percentage")

    parser.add_argument('--dataset', type=str, default='gc_10k')
    parser.add_argument('--kmeans_k', type=int, default=5)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--title', default='ssl_abmil_b1000_4*128_d10', type=str)

    # global args
    args = parser.parse_args()
    args.log_folder = os.path.join('output-model', args.title)
    
    world_size = torch.cuda.device_count()
    mp.spawn(ssl_training, args=(world_size, args, ), nprocs=world_size)
