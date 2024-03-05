import torch
from PIL import Image
import open_clip
from torchvision import transforms
import os
import time
from torch.utils.data import DataLoader, Dataset
import argparse
from PIL import Image
import h5py
import glob

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file1 = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file1:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file1.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file1[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file1.close()
    return output_path

# patchs in one bag 
class Whole_Slide_Patchs_Ngc(Dataset):
    def __init__(self,
                 wsi_path,
                 target_patch_size,
                 preprocess):
        # for resnet50
        self.preprocess = transforms.Compose([
            transforms.Resize(target_patch_size),
            transforms.ToTensor(),
        ])
        # for biomedclip
        if preprocess != None:
            self.preprocess = preprocess
        self.wsi_path = wsi_path
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        self.patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
                int(os.path.basename(x).split(".")[0].split("_")[1])))
        
    def __getitem__(self, idx):
        img = Image.open(self.patch_files[idx])
        img = self.preprocess(img)
        return img, os.path.basename(self.patch_files[idx])
    
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
    dataset = Whole_Slide_Patchs_Ngc(wsi_dir, target_patch_size, preprocess_val)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    mode = 'w'
    for i, (batch, wsi_names) in enumerate(loader):
        print(i, wsi_names)
        with torch.no_grad():
            if i % print_every == 0:
                print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
            batch = batch.to(device)
            features, text_features, logit_scale = model(batch)
            features = features.cpu().numpy()
            
            asset_dict = {'features': features}
            save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
            mode = 'a'
    return output_path
    
def main():
    # set argsrget_patch
    parser = argparse.ArgumentParser(description='NGC dataset Feature Extraction')
    parser.add_argument('--wsi_root', type=str, default='/home1/wsi')
    parser.add_argument('--output_path', type=str, default='result-final-ngc-features')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--batch_size', type=int, default=16)  
    parser.add_argument('--num_workers', type=int, default=16)  
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=(224, 224))
    # model options
    parser.add_argument('--ckp_path', type=str, default=None)
    args = parser.parse_args()
    
    # get wsi paths
    wsi_root = args.wsi_root
    wsi_dirs = ["ngc-2023-1333/Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM/BJFFK-ZS571537"]
    wsi_dirs = [os.path.join(wsi_root, d) for d in wsi_dirs]
    
    # get output path
    output_path = args.output_path
    output_path_pt = os.path.join(output_path, 'pt')
    output_path_h5 = os.path.join(output_path, 'h5_files')
    os.makedirs(output_path_pt, exist_ok=True)
    os.makedirs(output_path_h5, exist_ok=True)
    dest_files = os.listdir(output_path_pt)
    
    print('loading model')
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

    model = model.to(device)
    model.eval()
    total = len(wsi_dirs)    

    for idx in range(total):
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
        
        file1 = h5py.File(output_file_path, 'r')
        
        features = file1['features'][:]
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