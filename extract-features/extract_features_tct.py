import torch
import torch.nn as nn
from math import floor
from PIL import Image
import open_clip
from torchvision import transforms
import os
import copy
import time
from torch.utils.data import DataLoader, Dataset
from models.model_adapter import LinearAdapter
from models.model_backbone import ResnetBackbone, BiomedclipBackbone, ClipBackbone, PlipBackbone
import argparse
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# patchs in one bag 
class Whole_Slide_Patchs_Ngc(Dataset):
    def __init__(self,
                 wsi_path,
                 target_patch_size,
                 preprocess,
                 is_plip=False):
        # for resnet50
        self.preprocess = transforms.Compose([
            transforms.Resize(target_patch_size),
            transforms.ToTensor(),
        ])
        # for plip
        self.is_plip = is_plip
        # for biomedclip
        if preprocess != None:
            self.preprocess = preprocess
        self.wsi_path = wsi_path
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        # sort to ensure reproducibility
        try:
            self.patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        except Exception as e:
            print(e)
            
    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx])
        if self.is_plip:
            img = self.preprocess(images=img, return_tensors='pt').data['pixel_values'].squeeze(0)
        else:
            img = self.preprocess(img)
        return img
    
    def __len__(self):
        return len(self.patch_files)

    def __str__(self) -> str:
        return f'the length of patchs in {self.wsi_path} is {self.__len__()}'

def compute_w_loader(wsi_dir, 
                      output_path, 
                      model,
                      preprocess_val, # for biomedclip pretrain
                      args):
    # set parameters
    batch_size = args.batch_size
    target_patch_size = args.target_patch_size
    num_workers = args.num_workers
    verbose = args.verbose
    print_every = args.print_every
    if args.base_model == 'plip' and not args.default_preprocess:
        dataset = Whole_Slide_Patchs_Ngc(wsi_dir, target_patch_size, preprocess_val, is_plip=True)
    else:
        dataset = Whole_Slide_Patchs_Ngc(wsi_dir, target_patch_size, preprocess_val)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    mode = 'w'
    for i, batch in enumerate(loader):
        with torch.no_grad():
            if i % print_every == 0:
                print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
            batch = batch.to(device)
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
            features = features.cpu().numpy()
            asset_dict = {'features': features}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path
    
def main():
    # set argsrget_patch
    parser = argparse.ArgumentParser(description='NGC dataset Feature Extraction')
    parser.add_argument('--dataset', type=str, default='gc', choices=['ngc', 'ubc', 'gc', 'fnac'])
    parser.add_argument('--wsi_root', type=str, default='/home1/wsi/gc-224')
    parser.add_argument('--output_path', type=str, default='result-final-gc-features')
    parser.add_argument('--feat_dir', type=str, default='resnet-ori-test')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--num_workers', type=int, default=16)  
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=(224, 224))
    # model options
    parser.add_argument('--base_model', default='resnet50', type=str, choices=['biomedclip', 'resnet50', 'resnet34', 'resnet18', 'plip', 'clip'])
    parser.add_argument('--with_adapter', action='store_true')
    parser.add_argument('--ckp_path', type=str, default=None)
    parser.add_argument('--without_head', action='store_true')
    parser.add_argument('--default_preprocess', action='store_true')
    args = parser.parse_args()
    
    if args.multi_gpu:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    
    # get wsi paths
    wsi_root = args.wsi_root
    if args.dataset == 'ngc':
        sub_paths = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM',
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        data_roots = list(map(lambda x: os.path.join(wsi_root, x), sub_paths)) 
        wsi_dirs = []

        for data_root in data_roots:
            wsi_dirs.extend([os.path.join(data_root, subdir) for subdir in os.listdir(data_root)])
    
    elif args.dataset == 'gc':
        sub_paths = [
            'NILM',
            'POS'
        ]
        wsi_dirs = []
        for sub_path in sub_paths:
            wsi_dirs.extend([os.path.join(wsi_root, sub_path, wsi_name) for wsi_name in os.listdir(os.path.join(wsi_root, sub_path))])
            
    elif args.dataset == 'ubc':
        wsi_dirs = [os.path.join(wsi_root, subdir) for subdir in os.listdir(wsi_root)]
    
    elif args.dataset == 'fnac':
        wsi_dirs = [os.path.join(wsi_root, subdir) for subdir in os.listdir(wsi_root)]
        
    
    # get output path
    output_path = args.output_path
    output_path = os.path.join(output_path, args.feat_dir)
    output_path_pt = os.path.join(output_path, 'pt')
    output_path_h5 = os.path.join(output_path, 'h5_files')
    os.makedirs(output_path_pt, exist_ok=True)
    os.makedirs(output_path_h5, exist_ok=True)
    dest_files = os.listdir(output_path_pt)
    
    # load model
    torch.cuda.set_device(args.local_rank)
    print('loading model')
    preprocess_val = None
    if args.base_model == 'resnet50':
        backbone = ResnetBackbone(pretrained=True, name='resnet50')
        input_dim = 1024
    elif args.base_model == 'resnet34':
        backbone = ResnetBackbone(pretrained=True, name='resnet34')
        input_dim = 512
    elif args.base_model == 'resnet18':
        backbone = ResnetBackbone(pretrained=True, name='resnet18')
        input_dim = 512
    elif args.base_model == 'biomedclip':
        backbone = BiomedclipBackbone(args.without_head)
        input_dim = 512
        if args.without_head:
            input_dim = 768
    elif args.base_model == 'clip':
        backbone = ClipBackbone()
        input_dim = 512
    elif args.base_model == 'plip':
        backbone = PlipBackbone()
        input_dim = 512
    preprocess_val = backbone.preprocess_val
    
    if args.default_preprocess:
        preprocess_val = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor()])
            
    print('load backbone successfully')
    
    if args.with_adapter:
        adapter = LinearAdapter(input_dim)
        ckp = torch.load(args.ckp_path)
        adapter.load_state_dict(ckp['adapter'])
        model = nn.Sequential(backbone, adapter).to(device)
    else:
        model = nn.Sequential(backbone).to(device)
    model.eval()
    total = len(wsi_dirs)    

    for idx in range(total):
        if idx % args.world_size != args.local_rank:
            continue
        
        wsi_dir = wsi_dirs[idx]
        wsi_name = os.path.basename(wsi_dir)
        print('\nprogress: {}/{}'.format(idx, total))
        print(wsi_name)
        
        if wsi_name+'.pt' in dest_files:
            print('skipped {}'.format(wsi_name))
            continue
        
        output_file_path = os.path.join(output_path_h5, wsi_name+'.h5')
        time_start = time.time()
        compute_w_loader(wsi_dir,
                         output_file_path,
                         model,
                         preprocess_val,
                         args,
                        )
        time_elapesd = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapesd))
        
        file = h5py.File(output_file_path, 'r')
        
        features = file['features'][:]
        print('features size: ', features.shape)
        features = torch.from_numpy(features)
        torch.save(features, os.path.join(output_path_pt, wsi_name+'.pt'))

if __name__ == '__main__':
    time_start = time.time()
    
    main()
    
    time_end = time.time()
    time_elapesd = time_end - time_start
    print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                    time_elapesd%3600//60,
                                                    time_elapesd%60))