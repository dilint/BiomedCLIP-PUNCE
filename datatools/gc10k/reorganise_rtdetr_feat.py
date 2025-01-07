import torch
import numpy as np
import os, time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def cell2patch(wsi_name, index):
    cell_names = os.listdir(os.path.join(root_path, wsi_name))
    # cell_name = cell_names[0]
    # print(cell_name.split('-')[-1])
    if os.path.exists(os.path.join(output_path, wsi_name+'.pt')): # 跳过被处理过的WSI数据
        time.sleep(0.5)
        print('[{}/{}] {} has been saved, total {} patches'.format(index, len(wsi_names), wsi_name, len(patch_embeds)))
        return 
    feat_per_patch = {}
    for cell_name in cell_names:
        tmp = cell_name.split('_')
        patch_name = tmp[-1]
        cell_cat = tmp[0]
        cell_embed = np.load(os.path.join(root_path, wsi_name, cell_name))
        if patch_name not in feat_per_patch.keys():
            feat_per_patch[patch_name] = [cell_embed]
        else:
            feat_per_patch[patch_name].append(cell_embed)
    patch_embeds = []
    for patch_name in feat_per_patch.keys():
        patch_embed = np.stack(feat_per_patch[patch_name]) # [M, 256]
        patch_embeds.append(patch_embed) # [N, M~, 256]
    torch.save(patch_embeds, os.path.join(output_path, wsi_name+'.pt'))
    print('[{}/{}] {} is saved successfully, total {} patches'.format(index, len(wsi_names), wsi_name, len(patch_embeds)))


root_path = '/mnt/tmp/TCT_DATA/rtdetr_feats_20240822_embed_x7/img_feats/embed6'
output_path = '/data/wsi/TCTGC10k-Cell/rtdetr-v1-2025.1.1'
# wsi_name = '1914834909-01-HSIL'
wsi_names = os.listdir(root_path)
indexs = torch.arange(len(wsi_names))

time_start = time.time()
with ThreadPoolExecutor(max_workers=20) as executor:
    # map方法会异步执行square函数，对numbers列表中的每个元素进行处理
    futures = [executor.submit(cell2patch, wsi_name, index) for wsi_name, index in zip(wsi_names, indexs)]
time_end = time.time()
time_elapesd = time_end - time_start
print('\n The program took {} h {} min {} s'.format(time_elapesd//3600,
                                                time_elapesd%3600//60,
                                                time_elapesd%60))