import torch
import torch.nn.functional as F
import cv2
import os
import time
import glob
import argparse
import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader, Dataset
from utils.utils import seed_torch

class Whole_Slide_Patchs(Dataset):
    def __init__(self, wsi_path):
        self.wsi_path = wsi_path
        self.patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + \
                           glob.glob(os.path.join(wsi_path, '*.png'))
        try:
            self.patch_files = sorted(self.patch_files, key=lambda x: (
                int(os.path.basename(x).split(".")[0].split("_")[0]), 
                int(os.path.basename(x).split(".")[0].split("_")[1])
            ))
        except Exception:
            self.patch_files.sort()

    def __getitem__(self, idx):
        img = cv2.imread(self.patch_files[idx])
        if img is None:
            return torch.zeros((3, 256, 256), dtype=torch.uint8)
        
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        return img_tensor

    def __len__(self):
        return len(self.patch_files)

def postprocess_feats(confidence_thres, outputs):
    wsi_part_feats = []
    box_cls = outputs[0]
    embed6 = outputs[-1]
    
    B, _, _ = box_cls.shape 
    for b in range(B): 
        max_scores = np.amax(box_cls[b][:, 4:], axis=1) 
        selected_indices = np.where(max_scores >= confidence_thres)[0]
        patch_feat = embed6[b][selected_indices]
        
        if patch_feat.shape[0] == 0:
            continue
        wsi_part_feats.append(torch.from_numpy(patch_feat))
        
    return wsi_part_feats

def my_collate_fn(batch):
    return torch.stack(batch, dim=0)

def compute_w_loader(wsi_dir, output_path, session, target_patch_size, args):
    batch_size = args.batch_size
    
    dataset = Whole_Slide_Patchs(wsi_dir)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=args.num_workers, 
                        collate_fn=my_collate_fn, pin_memory=True)
    
    if args.verbose > 0:
        print(f'processing {wsi_dir}: total of {len(loader)} batches')
    
    wsi_feats = []
    
    device_id = args.device_ids[args.local_rank]
    device = torch.device(f'cuda:{device_id}')
    
    input_name = session.get_inputs()[0].name
    output_names = [node.name for node in session.get_outputs()]

    for i, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        batch = batch.float().div(255.0)
        batch = batch[:, [2, 1, 0], :, :]
        
        batch = F.interpolate(batch, size=tuple(target_patch_size), 
                              mode='bilinear', align_corners=False)
        
        if not batch.is_contiguous():
            batch = batch.contiguous()

        io_binding = session.io_binding()
        
        io_binding.bind_input(
            name=input_name,
            device_type='cuda',
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(batch.shape),
            buffer_ptr=batch.data_ptr()
        )
        
        for name in output_names:
            io_binding.bind_output(name, 'cuda', device_id=device_id)
        
        session.run_with_iobinding(io_binding)
        
        ort_outputs = io_binding.get_outputs()
        np_outputs = [out.numpy() for out in ort_outputs]
        
        if i % args.print_every == 0:
            print(f'batch {i}/{len(loader)}, processed {i * batch_size} files')
            
        wsi_part_feats = postprocess_feats(args.confidence_thres, np_outputs)
        wsi_feats.extend(wsi_part_feats)
    
    n = len(wsi_feats)
    if n > 0:
        m_max = max([f.shape[0] for f in wsi_feats]) if n > 0 else 0
        count_m = sum([f.shape[0] for f in wsi_feats])
        print(f'[{os.path.basename(wsi_dir)}] patch num: {n}, max cell: {m_max}, mean cell: {count_m/n:.2f}, zero patches: {len(dataset)-n}')
    
    torch.save(wsi_feats, output_path)
    return output_path
    
def main():
    parser = argparse.ArgumentParser(description='NGC dataset Feature Extraction')
    parser.add_argument('--dataset', type=str, default='gc2625', choices=['ngc', 'ubc', 'gc2625', 'fnac', 'gc'])
    parser.add_argument('--wsi_root', type=str, default='/data/wsi/TCTGC2625/gc')
    parser.add_argument('--output_path', type=str, default='/data/wsi/TCT2625-features')
    parser.add_argument('--feat_dir', type=str, default='rtdetr')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=20)
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=[1280, 1280])
    parser.add_argument('--model_path', type=str, default='/home/huangjialong/projects/BiomedCLIP-PUNCE/PatchEncoder/best_x7_20240822.onnx')
    parser.add_argument('--device_ids', type=int, nargs='+', default=(0,1,2,3))
    parser.add_argument('--confidence_thres', type=float, default=0.3)
    args = parser.parse_args()
    
    if args.multi_gpu:
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    wsi_dirs = []
    if args.dataset == 'ngc':
        sub_paths = [
            'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-NILM', 'Unannotated_KSJ/Unannotated-KSJ-TCTNGC-POS',
            'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-NILM', 'Unannotated_XIMEA/Unannotated-XIMEA-TCTNGC-POS'
        ]
        for sp in sub_paths:
            full_sp = os.path.join(args.wsi_root, sp)
            wsi_dirs.extend([os.path.join(full_sp, d) for d in os.listdir(full_sp)])
    elif args.dataset == 'gc2625':
        for sp in ['NILM', 'POS']:
            full_sp = os.path.join(args.wsi_root, sp)
            wsi_dirs.extend([os.path.join(full_sp, d) for d in os.listdir(full_sp)])
    elif args.dataset == 'gc':
        wsi_dirs = [os.path.join(args.wsi_root, d) for d in os.listdir(args.wsi_root)]
        
    output_path = os.path.join(args.output_path, args.feat_dir, 'pt')
    os.makedirs(output_path, exist_ok=True)
    dest_files = set(os.listdir(output_path))
    
    print('Loading model...')
    providers = [('CUDAExecutionProvider', {'device_id': args.device_ids[args.local_rank]})]
    session = ort.InferenceSession(args.model_path, providers=providers)
    print('Load backbone successfully')
    
    total = len(wsi_dirs)
    
    for idx in range(total):
        if idx % args.world_size != args.local_rank:
            continue
            
        wsi_dir = wsi_dirs[idx]
        wsi_name = os.path.basename(wsi_dir)
        
        if wsi_name + '.pt' in dest_files:
            continue
            
        print(f'\nProcessing: {wsi_name} ({idx}/{total})')
        output_file_path = os.path.join(output_path, wsi_name + '.pt')
        
        start = time.time()
        compute_w_loader(wsi_dir, output_file_path, session, args.target_patch_size, args)
        print(f'Done. Took {time.time() - start:.2f} s')

if __name__ == '__main__':
    start_time = time.time()
    seed_torch(2024)
    main()
    elapsed = time.time() - start_time
    print(f'\nTotal time: {elapsed//3600:.0f}h {elapsed%3600//60:.0f}m {elapsed%60:.0f}s')