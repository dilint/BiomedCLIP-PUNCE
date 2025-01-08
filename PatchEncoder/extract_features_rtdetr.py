import torch
import torch.nn as nn
from PIL import Image
import cv2
from torchvision import transforms
import os
import time
from torch.utils.data import DataLoader, Dataset
import argparse
from utils.file_utils import save_hdf5
from PIL import Image
import glob
from utils.utils import seed_torch
import numpy as np
import onnxruntime as ort

class Whole_Slide_Patchs(Dataset):
    def __init__(self,
                 wsi_path,
                 target_patch_size):
        """
        参数:
            wsi_path (str): ,WSI的路径
            target_patch_size (tuple): (input_heght, input_weight), 输入到网络的patch的大小
        """
        self.wsi_path = wsi_path
        self.target_patch_size = target_patch_size
        patch_files = glob.glob(os.path.join(wsi_path, '*.jpg')) + glob.glob(os.path.join(wsi_path, '*.png'))
        # sort to ensure reproducibility
        try:
            self.patch_files = sorted(patch_files, key=lambda x: (int(os.path.basename(x).split(".")[0].split("_")[0]), 
        int(os.path.basename(x).split(".")[0].split("_")[1])))
        except Exception as e:
            print(e)
    
    def preprocess(self, img):
        """
        在执行推理之前预处理输入图像。
        参数:
            img (ndarray): [img.height, img.width, 3],cv2.imread()读取的图像
        返回:
            image_data (ndarray): [3, input_height, input_width] ,为推理准备好的预处理后的图像数据。
        """
        # img_height, img_width = img.shape[:2]
        input_height, input_width = self.target_patch_size

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_width, input_height))
        image_data = np.array(img) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # 通道调整
        # image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        return image_data
        
    def __getitem__(self, idx):
        img = cv2.imread(self.patch_files[idx]).astype(np.float32)
        img = self.preprocess(img)
        return img
    
    def __len__(self):
        return len(self.patch_files)

    def __str__(self) -> str:
        return f'the length of patchs in {self.wsi_path} is {self.__len__()}'

def postprocess_feats(confidence_thres, outputs):
    """
    对模型输出进行后处理

    参数:
        confidence_thres (float): 0.3, 置信度。
        output (list): [scores, embed1,..., embed6];box_cls([B,300,17]);embed6([B, 300, 256]),模型的输出，其中box_cls为box的4个信息和13类别得分。

    返回:
        wsi_part_feats (list): [[M1,256], [M2,256], ...[Mn,256]], 其中[M,256]为ndarray;批量图片的特征
    """
    wsi_part_feats = []
    
    box_cls = outputs[0]
    embed6 = outputs[-1]
    
    B, _, _ = box_cls.shape # [B, 300, 17]
    for b in range(B): # box_cls[b]: [300, 17], embed6[b]: [300, 256]
        patch_feat = []
        max_scores = np.amax(box_cls[b][:, 4:], axis=1) # [300]
        # 进行阈值筛选
        selected_indices = np.where(max_scores >= confidence_thres)[0] # 选中阈值
        patch_feat = embed6[b][selected_indices]
        if patch_feat.shape[0] == 0:
            continue
        wsi_part_feats.append(torch.from_numpy(patch_feat))
        
    # b张图片的检测特征信息
    return wsi_part_feats

def my_collate_fn(batch):
    batch = np.stack(batch, axis=0)
    return batch

def compute_w_loader(wsi_dir, 
                      output_path, 
                      session,
                      target_patch_size, # for biomedclip pretrain
                      args):
    # set parameters
    batch_size = args.batch_size
    num_workers = args.num_workers
    verbose = args.verbose
    print_every = args.print_every

    dataset = Whole_Slide_Patchs(wsi_dir, target_patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=my_collate_fn)
    if verbose > 0:
        print('processing {}: total of {} batches'.format(wsi_dir,len(loader)))
    
    wsi_feats = []    
    for i, batch in enumerate(loader):
        with torch.no_grad():
            if i % print_every == 0:
                print('batch {}/{}, {} files processed'.format(i, len(loader), i * batch_size))
            input_name = session.get_inputs()[0].name
            # print(type(batch), batch[0].dtype, batch.shape)
            outputs = session.run(None, {input_name: batch})
            wsi_part_feats = postprocess_feats(args.confidence_thres, outputs)
            wsi_feats.extend(wsi_part_feats)
    
    # 统计patch数量和单个patch的最大细胞数量
    n, m_max, count_m, count_0 = len(wsi_feats), 0, 0, 0
    for feature in wsi_feats:
            m_now = feature.shape[0]
            if m_now > m_max:
                m_max = m_now
            if m_now == 0:
                count_0 += 1
            count_m += m_now
    print(f'[{os.path.basename(wsi_dir)}] patch num & max cell num & mean cell num & zero patch num: ', n, m_max, m_now/n, count_0)
    torch.save(wsi_feats, output_path)
    
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
    parser.add_argument('--target_patch_size', type=int, nargs='+', default=(1280, 1280))
    # model options
    parser.add_argument('--model_path', type=str, default='/home/huangjialong/projects/tctcls-lp/det-ljx/best_x7_20240822.onnx')
    parser.add_argument('--device_ids', type=int, nargs='+', default=(1,2))
    parser.add_argument('--confidence_thres', type=float, default=0.3, help='confidence threshold for detection feature extraction')  
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
    os.makedirs(output_path_pt, exist_ok=True)
    dest_files = os.listdir(output_path_pt)
    
    # 从onnx load model,几张卡就load几个
    print('loading model')
    
    # 判断是使用GPU或CPU
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': args.device_ids[args.local_rank],  # 根据对应位置选择gpu号
        }),
    #     'CPUExecutionProvider',  # 也可以设置CPU作为备选
    ]
    session = ort.InferenceSession(args.model_path, providers=providers)
    print('load backbone successfully')
    # 加载
    
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
        
        output_file_path = os.path.join(output_path_pt, wsi_name+'.pt')
        time_start = time.time()
        compute_w_loader(wsi_dir,
                         output_file_path,
                         session,
                         args.target_patch_size,
                         args,
                        )
        time_elapesd = time.time() - time_start
        print('\ncomputing features for {} took {} s'.format(output_path, time_elapesd))


if __name__ == '__main__':
    time_start = time.time()
    seed_torch(2024)
    main()
    
    time_end = time.time()
    time_elapesd = time_end - time_start
    print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                    time_elapesd%3600//60,
                                                    time_elapesd%60))