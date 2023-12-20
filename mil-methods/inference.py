import argparse, os
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils import *
from dataloader import *
from modules import mean_max

class WsiPathUtil():
    def __init__(self, wsi_root, output_root):
        sub_paths = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM',
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        self.wsi_root = wsi_root
        self.output_root = output_root
        self.wsi_dict = {}
        data_roots = list(map(lambda x: os.path.join(wsi_root, x), sub_paths)) 
        for data_root in data_roots: 
            for subdir in os.listdir(data_root):
                self.wsi_dict[subdir] = os.path.join(data_root, subdir)

    def saveSubimages(self, wsi_name, topk_dices):
        wsi_path = self.wsi_dict[wsi_name]
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg'))
        for dice in topk_dices:
            source_path = patch_files[dice]
            rel_path = os.path.relpath(source_path, self.wsi_root)
            dest_path = os.path.join(self.output_root, rel_path)
            if not os.path.exists(os.path.dirname(dest_path)):
                os.makedirs(os.path.dirname(dest_path))
            os.system('cp {} {}'.format(source_path, dest_path))

def topk(tensor, k):
    if tensor.size()[0] < k:
        k = tensor.size()[0]
    values, indices = torch.topk(tensor, k, dim=0, largest=True, sorted=False)
    return indices


def main(args):
    # set seed
    seed_torch(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    topk_num = args.topk_num
    # get dataset
    wsi_root = args.wsi_root
    feature_root = args.feature_root
    train_label = args.train_label
    output_root = args.output_root
    if args.datasets.lower() == 'ngc':
        dataset_p, dataset_l = [], []
        with open(train_label, 'r') as file:
            lines = file.readlines()
            for line in lines:
                dataset_p.append(line.split(',')[0])
                dataset_l.append(line.split(',')[1])
            file.close()
        dataset_p = np.array(dataset_p)
        dataset_l = np.array(dataset_l)
        dataset = NGCDatasetInfer(dataset_p,dataset_l,feature_root)
        print(len(dataset))
        
    # build model
    model_type = args.model
    model_args = {
        'n_classes': args.n_classes,
        'dropout': args.dropout,
        'act': args.act,
        'input_dim': args.input_dim
    }
    if model_type == 'meanmil':
        model = mean_max.MeanMIL(test=True, **model_args).to(device)
    elif model_type == 'maxmil':
        model = mean_max.MaxMIL(test=True, **model_args).to(device)
    
    # load checkpoint
    ckp_path = args.ckp_path
    ckp = torch.load(ckp_path)
    if 'model' in ckp:
        ckp = ckp['model']
    model.load_state_dict(ckp)

    # inference
    wsi_util = WsiPathUtil(wsi_root, output_root)
    loader_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
    }
    loader = DataLoader(dataset, shuffle=False, **loader_args)
    
    with torch.no_grad():
        for data in tqdm(loader):
            bag = data[0].to(device)
            label = data[1].to(device)
            wsi_name = data[2][0]
            if model_type in ('mhim','pure'):
                test_logits = model.forward_test(bag)
            elif model_type == 'dsmil':
                test_logits, _ = model(bag)
            else:
                test_logits = model(bag)
            test_logits = test_logits.squeeze()
            test_logits = test_logits.cpu()[:, 1]
            topk_dices = topk(test_logits, topk_num)
            wsi_util.saveSubimages(wsi_name=wsi_name, topk_dices=topk_dices)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter high risk patches for ngc')
    
    # Dataset
    parser.add_argument('--datasets', default='ngc', type=str, help='[camelyon16, ngc]')
    parser.add_argument('--feature_root', default='/root/project/BiomedCLIP-PUNCE/extract-features/result-final-ngc-features/resnet_simclr_infonce_ngc_none_224_none', type=str, help='wsi feature root path')
    parser.add_argument('--train_label', default='/root/project/BiomedCLIP-PUNCE/mil-methods/ngc-labels/train_label.csv',type=str, help='the path of train wsi list')
    parser.add_argument('--wsi_root', default='/root/commonfile/wsi/ngc-2023-1333', type=str, help='Dataset root path')
    # Model
    parser.add_argument('--model', default='meanmil', type=str, help='[pure, mhim, meanmil, maxmil]')
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--dropout', default=0.25, type=float, help='Dropout in the projection head')
    parser.add_argument('--act', default='relu', type=str, help='Activation func in the projection head [gelu,relu]')
    parser.add_argument('--input_dim', default=1024, type=int, help='The dimention of patch feature')
    # Inference
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers in the dataloader')
    parser.add_argument('--batch_size', default=1, type=int, help='Number of batch size')
    
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--ckp_path', default='/root/project/BiomedCLIP-PUNCE/mil-methods/output-model/output-test/resnet1-meanmil-ngc/fold_0_model_best_auc.pt', type=str)
    # Output
    parser.add_argument('--output_root', default='/root/commonfile/wsi/output-filter/ngc-meanmil/', type=str, help='output path')
    parser.add_argument('--topk_num', default=50, type=int, help='topk_num')
    args = parser.parse_args()
    main(args)
    