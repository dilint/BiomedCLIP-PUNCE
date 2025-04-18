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
from models.model_backbone import ResnetBackbone, BiomedclipBackbone, ClipBackbone, PlipBackbone, Dinov2Backbone, GigapathBackbone, MaeBackbone, Virchow2Backbone, Uni2Backbone
import argparse
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import glob
from utils.utils import seed_torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Whole_Slide_Patchs(Dataset):
    def __init__(self,
                 wsi_path,
                 target_patch_size,
                 preprocess,
                 is_plip=False,
                 fine_grained=False,
                 fine_grained_pre_size=(1280,1280),
                 fine_grained_size=(256,256)):
        """
        wsi_path: WSI的路径
        target_patch_size: 输入到网络的patch的大小
        preprocess: foundation model自带的预处理函数，如果有的话，target_patch_size会被忽略
        is_plip: 是否使用plip作为foundation model, 因为plip需要特殊的预处理
        fine_grained: 是否对原始的patch先进行切割成更小的图片 
        fine_grained_size: 如果fine_grained为True，则这个参数为切割后的小图片的大小
        """
        # for resnet50
        self.preprocess = transforms.Compose([
            transforms.Resize(target_patch_size),
            transforms.ToTensor(),
        ])
        # for fine grained
        self.fine_grained = fine_grained
        self.fine_grained_size = fine_grained_size
        self.fine_grained_pre_size = fine_grained_pre_size
        
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
    
    def _crop_patchs(self, img):
        # crop patches from the image
        patches = []
        fine_grained_size = self.fine_grained_size
        for j in range(0, img.size[1], fine_grained_size[1]):
            for i in range(0, img.size[0], fine_grained_size[0]): 
                patch = img.crop((i, j, i + fine_grained_size[0], j + fine_grained_size[1]))
                patches.append(patch)
        return patches
    
    def __getitem__(self, idx):
        try:
            img = Image.open(self.patch_files[idx])
        except Exception as e:
            with open('/data/hjl/data/tmp/log.txt', 'a') as f:
                f.write(f'{e}\n')
            return None
        if self.fine_grained:
            img = transforms.Resize(self.fine_grained_pre_size)(img)
            imgs = self._crop_patchs(img)
        else:
            imgs = [img]
        if self.is_plip:
            imgs_process = torch.stack([self.preprocess(images=img, return_tensors='pt').data['pixel_values'].squeeze(0) for img in imgs]).squeeze(0)
        else:
            imgs_process = torch.stack([self.preprocess(img) for img in imgs]).squeeze(0)
        return imgs_process
    
    def __len__(self):
        return len(self.patch_files)

    def __str__(self) -> str:
        return f'the length of patchs in {self.wsi_path} is {self.__len__()}'

def my_collate(batch): 
    len_batch = len(batch)  # original batch length
    batch = list(filter(lambda x: x is not None, batch))  # filter out all the Nones
    if len(batch) == 0:  # if all the samples are None, return an empty batch
        return batch 
    # if len_batch > len(batch):  # source all the required samples from the original dataset at random
    #     diff = len_batch - len(batch)
    #     for i in range(diff):
    #         item = dataset[np.random.randint(0, len(dataset))]
    #         while item is None:
    #             item = dataset[np.random.randint(0, len(dataset))]
    #         batch.append(item)
    return torch.utils.data.dataloader.default_collate(batch) 

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
    is_plip = False
    if args.base_model == 'plip' and not args.default_preprocess:
        is_plip = True
    dataset = Whole_Slide_Patchs(wsi_dir, target_patch_size, preprocess_val, is_plip=is_plip,\
                                fine_grained=args.fine_grained, fine_grained_size=args.fine_grained_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=my_collate)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    mode = 'w'
    try:
        for i, batch in enumerate(loader):
            if args.only_load:
                return
            
            if len(batch) == 0:
                continue
            with torch.no_grad():
                if i % print_every == 0:
                    print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
                if args.fine_grained:
                    M = batch.shape[1]
                    # [N,M,3,224,224]->[N*M,3,224,224]
                    batch = batch.reshape(-1,batch.shape[2],batch.shape[3],batch.shape[4])
                batch = batch.to(device)
                features = model(batch)
                if isinstance(features, tuple):
                    features = features[0]
                if args.fine_grained:
                    # [N*M,C] -> [N,M,C]
                    features = features.reshape(-1, M,features.shape[1])
                features = features.cpu().numpy()
                asset_dict = {'features': features}
                save_hdf5(output_path, asset_dict, attr_dict=None, mode=mode)
                mode = 'a'
    except Exception as e:
        print('**** {} error {}'.format(wsi_dir, e))
        with open(f'/data/wsi/TCTGC50k/error_{args.local_rank}.txt', 'a') as f:
            f.write('**** {} error {}'.format(wsi_dir, e))
        return False
    return output_path
    
def main():
    # set argsrget_patch
    parser = argparse.ArgumentParser(description='NGC dataset Feature Extraction')
    parser.add_argument('--dataset', type=str, default='gc', choices=['ngc', 'ubc', 'gc2625', 'fnac', 'gc'])
    parser.add_argument('--wsi_root', type=str, default='/data/wsi/TCTGC50k/TCTGC50k-volume1')
    parser.add_argument('--output_path', type=str, default='/data/wsi/TCTGC50k-features')
    parser.add_argument('--feat_dir', type=str, default='gigapath')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    # inference options 
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=40)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=(224, 224))
    parser.add_argument('--fine_grained', type=int, default=0)
    parser.add_argument('--fine_grained_size', type=int, nargs='+', default=(256, 256))
    parser.add_argument('--fine_grained_pre_size', type=int, nargs='+', default=(1280, 1280))
    # model options
    parser.add_argument('--base_model', default='mae', type=str, choices=['biomedclip', 'resnet50', 'resnet34', 'resnet18', 'plip', 'clip', 'dinov2', 'gigapath', 'mae','virchow2','uni2','conch'])
    parser.add_argument('--with_adapter', action='store_true')
    parser.add_argument('--backbone_weight_path', type=str, default=None)
    parser.add_argument('--ckp_path', type=str, default=None)
    parser.add_argument('--without_head', action='store_true')
    parser.add_argument('--default_preprocess', action='store_true')
    parser.add_argument('--only_load', action='store_true')
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
    
    elif args.dataset == 'gc2625':
        sub_paths = [
            'NILM',
            'POS'
        ]
        wsi_dirs = []
        for sub_path in sub_paths:
            wsi_dirs.extend([os.path.join(wsi_root, sub_path, wsi_name) for wsi_name in os.listdir(os.path.join(wsi_root, sub_path))])
    
    elif args.dataset == 'gc':
        wsi_dirs = [os.path.join(wsi_root, subdir) for subdir in os.listdir(wsi_root)]

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
    input_dim = 512
    
    if args.base_model == 'resnet50':
        backbone = ResnetBackbone(pretrained=True, name='resnet50')
        input_dim = 1024
    elif args.base_model == 'resnet34':
        backbone = ResnetBackbone(pretrained=True, name='resnet34')
    elif args.base_model == 'resnet18':
        backbone = ResnetBackbone(pretrained=True, name='resnet18')
    elif args.base_model == 'biomedclip':
        backbone = BiomedclipBackbone(args.without_head)
        if args.without_head:
            input_dim = 768
    elif args.base_model == 'clip':
        backbone = ClipBackbone()
    elif args.base_model == 'plip':
        backbone = PlipBackbone()
    elif args.base_model == 'dinov2':
        backbone = Dinov2Backbone()
        input_dim = 768
    elif args.base_model == 'gigapath':
        backbone = GigapathBackbone()
        input_dim = 1536
    elif args.base_model == 'virchow2':
        backbone = Virchow2Backbone()
        input_dim = 2560
    elif args.base_model == 'uni2':
        backbone = Uni2Backbone()
        input_dim = 1536
    elif args.base_model == 'mae':
        backbone = MaeBackbone()
        input_dim = 768
    preprocess_val = backbone.preprocess_val
    
    # # 准备一批数据
    # batch_size = 32
    # input_data = torch.randn(batch_size, 3, 224, 224)
    # input_data = input_data.to(device)

    # # 预热
    # with torch.no_grad():
    #     for _ in range(10):
    #         _ = model(input_data)

    # # 测量时间
    # start_time = time.time()
    # with torch.no_grad():
    #     for _ in range(100):  # 进行100次前向传播
    #         _ = model(input_data)
    # end_time = time.time()

    # # 计算FPS
    # elapsed_time = end_time - start_time
    # fps = (100 * batch_size) / elapsed_time
    # print(f"{args.base_model}: 模型每秒处理 {fps} 个样本, 经过了{elapsed_time}秒")
    # return
    
    if args.default_preprocess:
        preprocess_val = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor()])
            
    print('load backbone successfully')
    
    if args.backbone_weight_path:
        backbone_ckp = torch.load(args.backbone_weight_path)
        info = backbone.load_state_dict(backbone_ckp['model'])
        print('backbone weight load:' + str(info))
        
    if args.with_adapter:
        adapter = LinearAdapter(input_dim)
        ckp = torch.load(args.ckp_path)
        info = adapter.load_state_dict(ckp['adapter'])
        print('adapter weight load:' + str(info))
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
        result = compute_w_loader(wsi_dir,
                         output_file_path,
                         model,
                         preprocess_val,
                         args,
                        )
        time_elapesd = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapesd))

        if not result:
            continue
        if args.only_load:
            continue
        
        file = h5py.File(output_file_path, 'r')
        
        features = file['features'][:]
        print('features size: ', features.shape)
        features = torch.from_numpy(features)
        torch.save(features, os.path.join(output_path_pt, wsi_name+'.pt'))

if __name__ == '__main__':
    time_start = time.time()
    seed_torch(2024)
    main()
    
    time_end = time.time()
    time_elapesd = time_end - time_start
    print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                    time_elapesd%3600//60,
                                                    time_elapesd%60))