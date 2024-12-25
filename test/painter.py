import os
from PIL import Image
from tqdm import tqdm
import pandas as pd

def painter_ori(data_root, wsi_name, img_width, img_height, resize_ratio, output_dir):
    wsi_dir = os.path.join(data_root, wsi_name)
    imgfiles = os.listdir(wsi_dir)
    xs, ys = [int(imgfile.split('.')[0].split('_')[0]) for imgfile in imgfiles], \
                [int(imgfile.split('.')[0].split('_')[1]) for imgfile in imgfiles]
    x_max, y_max = max(xs), max(ys)
    print('patch num: ({}, {}), total hight: {}, total width: {}'.format(x_max, y_max, x_max*4096, y_max*2816))
    print('total pixels: {} billion, total size: {} GB'.format(x_max * y_max * 4096 * 2816 / (10**9),\
                x_max * y_max * 4096 * 2816 * 3 / (1024**3))) 

    # 计算每个子图的大小
    patch_resize_x, patch_resize_y = img_width//resize_ratio, img_height//resize_ratio
    # 创建一个新的空白图像
    total_width = patch_resize_x * x_max
    total_height = patch_resize_y * y_max
    combined_img = Image.new('RGB', (total_width, total_height))

    # 将每张图像粘贴到新图像的适当位置
    for i, file in tqdm(enumerate(imgfiles)):
        img = Image.open(os.path.join(wsi_dir, file))
        img = img.resize((patch_resize_x, patch_resize_y))
        x, y = int(file.split('.')[0].split('_')[0]), int(file.split('.')[0].split('_')[1])
        left = (x-1) * patch_resize_x
        upper = (y-1) * patch_resize_y
        combined_img.paste(img, (left, upper))

    # 显示拼接后的图像
    combined_img.save(os.path.join(output_dir, wsi_name + '.jpg'))
    
if __name__ == '__main__':
    data_root = '/home1/wsi/gc-224/POS'
    img_width, img_height = 224, 224
    resize_ratio = 14
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output-ori')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    input_path = '/home/huangjialong/projects/BiomedCLIP-PUNCE/datatools/gc/n-labels/all_label.csv'
    df = pd.read_csv(input_path, header=None)
    labels = []
    hsil_wsis = []
    for item in df.values:
        wsi_name, label = item[0], item[1]
        if label == 4:
            hsil_wsis.append(wsi_name)
    # print(hsil_wsi)
    for hsil_wsi in hsil_wsis:
        painter_ori(data_root, hsil_wsi, img_width, img_height, resize_ratio, output_dir)