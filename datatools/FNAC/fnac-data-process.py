import os
import random
from PIL import Image
from tqdm import tqdm

import json

def main():
    """主函数"""
    
    # 读取配置文件中的根目录
    DATA_ROOT = "D:/Dataset/"
    current_file = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件的父目录
    file_name = current_file + '/../../settings/environment.json'
    if os.path.exists(file_name):
        with open(file_name, 'r') as file:
            info = json.load(file)
            DATA_ROOT = info["FNAC_DATA_ROOT"]
            
    # 定义各种路径 
    DATA_PATH = DATA_ROOT+'FNAC-2019/'
    DATA_PREFIXS = ["B", "M"]
    OUTPUT_PATH = DATA_ROOT+'FNAC-CROP/base-data/'   
      
    # 开始分割
    for data_prefix in DATA_PREFIXS:
        input_path = os.path.join(DATA_PATH, data_prefix)
        output_path = os.path.join(OUTPUT_PATH, data_prefix)
        splitImagesInDirectory(input_path, output_path)


def splitImage(image_path, output_path):
    
    image = Image.open(image_path)
    wsi_name = os.path.splitext(os.path.basename(image_path))[0]
    wsi_name = os.path.basename(os.path.dirname(image_path)) + "_" + wsi_name
    output_path = os.path.join(output_path, wsi_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    width, height = image.size
    overlap_rate = 0.5 
    crop_size = 256
    overlap = int(crop_size * overlap_rate)

    # 计算水平和垂直方向上的切割数量
    num_horizontal = (width - overlap) // (crop_size - overlap)
    num_vertical = (height - overlap) // (crop_size - overlap)


    for i in range(num_horizontal):
        for j in range(num_vertical):
            left = i * (crop_size - overlap)
            upper = j * (crop_size - overlap)
            right = left + crop_size
            lower = upper + crop_size

            # 切割图像
            crop = image.crop((left, upper, right, lower))
            tmp_output_path = os.path.join(output_path, f"{i}_{j}.jpg")
            crop.save(tmp_output_path)


def splitImagesInDirectory(path, output_path):
    # 遍历目录中的所有文件
    for filename in tqdm(os.listdir(path)):
        # 检查文件是否为图片
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            splitImage(image_path, output_path)


if __name__ == "__main__":
    main()