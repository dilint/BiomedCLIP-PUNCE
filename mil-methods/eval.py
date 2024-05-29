import torch
from timm.utils import AverageMeter,dispatch_clip_grad
from utils import *
import argparse, os
from torch.utils.data import DataLoader, RandomSampler
from modules import attmil,clam,mhim,dsmil,transmil,mean_max
from dataloader import *

def main(args):
    torch.backends.cudnn.enabled = False
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # set seed
    seed_torch(args.seed)
    # --->get dataset
    def parse_tct_dataset(args, file_name):
        test_label_path = os.path.join(args.label_path, file_name)
        p, l= [], []
        with open(test_label_path, 'r') as f:
            for line in f.readlines():
                p.append(line.split(',')[0])
                l.append(line.split(',')[1])
        p = [np.array(p)]
        l = [np.array(l)]
        return p, l
    
    k = 0
    if args.datasets.lower() == 'tct':
        test_p, test_l = parse_tct_dataset(args, 'test_label.csv')
        test_set = C16Dataset(test_p[k],test_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if args.c_h:
            test_c_p, test_c_l = parse_tct_dataset(args, 'test_c.csv')
            test_c_set = C16Dataset(test_c_p[k],test_c_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            test_c_loader = DataLoader(test_c_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
            test_h_p, test_h_l = parse_tct_dataset(args, 'test_h.csv')
            test_h_set = C16Dataset(test_h_p[k],test_h_l[k],root=args.dataset_root,persistence=args.persistence,keep_same_psize=args.same_psize)
            test_h_loader = DataLoader(test_h_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    else:
        test_loader = None
        print('##info##: 暂时只支持tct数据集，tct为通用的tct-ngc和tct-gc')        
        return
        
    if args.model == 'mhim':
        mrh_sche = None
        model_params = {
            'input_dim': args.input_dim,
            'baseline': args.baseline,
            'dropout': args.dropout,
            'mask_ratio' : args.mask_ratio,
            'n_classes': args.n_classes,
            'temp_t': args.temp_t,
            'act': args.act,
            'head': args.n_heads,
            'msa_fusion': args.msa_fusion,
            'mask_ratio_h': args.mask_ratio_h,
            'mask_ratio_hr': args.mask_ratio_hr,
            'mask_ratio_l': args.mask_ratio_l,
            'mrh_sche': mrh_sche,
            'da_act': args.da_act,
            'attn_layer': args.attn_layer,
        }
        
        model = mhim.MHIM(**model_params).to(device)
            
    elif args.model == 'pure':
        model = mhim.MHIM(input_dim=args.input_dim,
                          select_mask=False,
                          n_classes=args.n_classes,
                          act=args.act,
                          head=args.n_heads,
                          da_act=args.da_act,
                          baseline=args.baseline).to(device)
    elif args.model == 'attmil':
        model = attmil.DAttention(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'gattmil':
        model = attmil.AttentionGated(dropout=args.dropout).to(device)
    # follow the official code
    # ref: https://github.com/mahmoodlab/CLAM
    elif args.model == 'clam_sb':
        model = clam.CLAM_SB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'clam_mb':
        model = clam.CLAM_MB(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'transmil':
        model = transmil.TransMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
    elif args.model == 'dsmil':
        model = dsmil.MILNet(n_classes=args.n_classes,dropout=args.dropout,act=args.act).to(device)
        args.cls_alpha = 0.5
        args.cl_alpha = 0.5
        state_dict_weights = torch.load('./modules/init_cpk/dsmil_init.pth')
        info = model.load_state_dict(state_dict_weights, strict=False)
        if not args.no_log:
            print(info)
    elif args.model == 'meanmil':
        model = mean_max.MeanMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act,input_dim=args.input_dim).to(device)
    elif args.model == 'maxmil':
        model = mean_max.MaxMIL(n_classes=args.n_classes,dropout=args.dropout,act=args.act,input_dim=args.input_dim).to(device)

    best_std = torch.load(os.path.join(args.ckp_path, 'fold_{fold}_model_best_auc.pt'.format(fold=k)))
    info = model.load_state_dict(best_std['model'])
    print(info)
    accuracy, auc_value, precision, recall, specificity, fscore = test_loop(args,model,test_loader,device,False)
    print(f'''
          accuracy: {accuracy},
          auc_value: {auc_value},
          precision: {precision},
          recall: {recall},
          specificity: {specificity},
          fscore: {fscore}''')
    if args.c_h:
        c_h = True
        sen_c = test_loop(args,model,test_c_loader,device, c_h)
        sen_h = test_loop(args,model,test_h_loader,device, c_h)
        print(f'sen_c: {sen_c},\n sen_h: {sen_h}')
        
def test_loop(args,model,loader,device,c_h):
    model.eval()
    bag_logits, bag_labels=[], []

    with torch.no_grad():
        for i, data in enumerate(loader):
            if len(data[1]) > 1:
                bag_labels.extend(data[1].tolist())
            else:
                bag_labels.append(data[1].item())

            if isinstance(data[0],(list,tuple)):
                for i in range(len(data[0])):
                    data[0][i] = data[0][i].to(device)
                bag=data[0]
                batch_size=data[0][0].size(0)
            else:
                bag=data[0].to(device)  # b*n*1024
                batch_size=bag.size(0)

            label=data[1].to(device)
            if args.model in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif args.model == 'dsmil':
                test_logits,_ = model(bag)
            else:
                test_logits = model(bag)
                
            if args.loss == 'ce':
                if (args.model == 'dsmil' and args.ds_average) or (args.model == 'mhim' and isinstance(test_logits,(list,tuple))):
                    bag_logits.append((0.5*torch.softmax(test_logits[1],dim=-1)+0.5*torch.softmax(test_logits[0],dim=-1))[:,1].cpu().squeeze().numpy())
                else:
                    if args.n_classes == 2:
                        bag_logits.extend(torch.softmax(test_logits,dim=-1)[:,1].cpu().numpy())
                    else:
                        bag_logits.extend(torch.softmax(test_logits, dim=-1).cpu().numpy())
            # TODO have not updated            
            elif args.loss == 'bce':
                if args.model == 'dsmil' and args.ds_average:
                    bag_logits.append((0.5*torch.sigmoid(test_logits[1])+0.5*torch.sigmoid(test_logits[0]).cpu().squeeze().numpy()))
                else:
                    bag_logits.append(torch.sigmoid(test_logits).cpu().squeeze().numpy())

    # save the log file
    if not c_h:
        if args.n_classes == 2:
            accuracy, auc_value, precision, recall, specificity, fscore = six_scores(bag_labels, bag_logits, args.threshold)
            
            if args.output_auc:
                fpr, tpr, threshold = roc_curve(bag_labels, bag_logits, pos_label=1)
                auc_value = roc_auc_score(bag_labels, bag_logits)
                roc_output_dir = 'output_roc'
                feat_dir = args.dataset_root.split('/')[-1]
                roc_par_dir = os.path.join(roc_output_dir, feat_dir)
                if not os.path.exists(roc_par_dir):
                    # 使用os.makedirs递归创建目录
                    os.makedirs(roc_par_dir)
                roc_output_fpr_path = os.path.join(roc_par_dir, 'fpr.npy')
                roc_output_tpr_path = os.path.join(roc_par_dir, 'tpr.npy')
                roc_output_auc_path = os.path.join(roc_par_dir, 'auc.npy')
                np.save(roc_output_fpr_path, fpr)
                np.save(roc_output_tpr_path, tpr)
                np.save(roc_output_auc_path, auc_value)
                
        else:
            specificity = 0
            auc_value, accuracy, recall, precision, fscore = multi_class_scores(bag_labels, bag_logits)
        return accuracy, auc_value, precision, recall, specificity, fscore
    else: 
        sens = tct_recall(bag_labels, bag_logits, args.threshold)
        return sens
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MIL Training Script')
    
    # Dataset 
    parser.add_argument('--datasets', default='tct', type=str, help='[camelyon16, tcga, tct]')
    parser.add_argument('--dataset_root', default='extract-features/result-final-gc-features/biomed1', type=str, help='Dataset root path')
    parser.add_argument('--label_path', default='datatools/tct-gc/labels', type=str, help='label of train dataset')
    parser.add_argument('--fix_loader_random', action='store_true', help='Fix random seed of dataloader')
    parser.add_argument('--fix_train_random', action='store_true', help='Fix random seed of Training')
    parser.add_argument('--persistence', action='store_true', help='Load data into memory') 
    parser.add_argument('--same_psize', default=0, type=int, help='Keep the same size of all patches [0]')

    # Model
    # Other models
    parser.add_argument('--ds_average', action='store_true', help='DSMIL hyperparameter')
    # Our
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    parser.add_argument('--loss', default='ce', type=str, help='Classification Loss [ce, bce]')
    parser.add_argument('--n_classes', default=2, type=int, help='Number of classes')
    parser.add_argument('--model', default='pure', type=str, help='Model name')
    parser.add_argument('--seed', default=2024, type=int, help='random number [2021]' )
    parser.add_argument('--baseline', default='attn', type=str, help='Baselin model [attn,selfattn]')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--n_heads', default=8, type=int, help='Number of head in the MSA')
    parser.add_argument('--da_act', default='relu', type=str, help='Activation func in the DAttention [gelu,relu]')
    parser.add_argument('--input_dim', default=512, type=int, help='The dimention of patch feature')

    # Shuffle
    parser.add_argument('--patch_shuffle', action='store_true', help='2-D group shuffle')
    parser.add_argument('--group_shuffle', action='store_true', help='Group shuffle')
    parser.add_argument('--shuffle_group', default=0, type=int, help='Number of the shuffle group')

    # MHIM
    # Mask ratio
    parser.add_argument('--mask_ratio', default=0., type=float, help='Random mask ratio')
    parser.add_argument('--mask_ratio_l', default=0., type=float, help='Low attention mask ratio')
    parser.add_argument('--mask_ratio_h', default=0., type=float, help='High attention mask ratio')
    parser.add_argument('--mask_ratio_hr', default=1., type=float, help='Randomly high attention mask ratio')
    parser.add_argument('--mrh_sche', action='store_true', help='Decay of HAM')
    parser.add_argument('--msa_fusion', default='vote', type=str, help='[mean,vote]')
    parser.add_argument('--attn_layer', default=0, type=int)
    
    # Siamese framework
    parser.add_argument('--cl_alpha', default=0., type=float, help='Auxiliary loss alpha')
    parser.add_argument('--temp_t', default=0.1, type=float, help='Temperature')
    parser.add_argument('--teacher_init', default='none', type=str, help='Path to initial teacher model')
    parser.add_argument('--no_tea_init', action='store_true', help='Without teacher initialization')
    parser.add_argument('--init_stu_type', default='none', type=str, help='Student initialization [none,fc,all]')
    parser.add_argument('--tea_type', default='none', type=str, help='[none,same]')
    parser.add_argument('--mm', default=0.9999, type=float, help='Ema decay [0.9997]')
    parser.add_argument('--mm_final', default=1., type=float, help='Final ema decay [1.]')
    parser.add_argument('--mm_sche', action='store_true', help='Cosine schedule of ema decay')
    
    # Misc
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--threshold', default=0, type=float, help='the threshold of classification')
    parser.add_argument('--no_log', action='store_true', help='Without log')
    parser.add_argument('--c_h', action='store_true')
    parser.add_argument('--output_auc', action='store_true')
    parser.add_argument('--ckp_path', type=str, default='mil-methods/output-model/mil-methods/biomed1-meanmil-tct-trainval', help='Checkpoint path')
    
    args = parser.parse_args()
    main(args)