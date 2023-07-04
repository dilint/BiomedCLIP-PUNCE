import os
import random
from PIL import Image
from tqdm import tqdm


DATA_PATH = 'C:\\Users\\11030\\Desktop\\FNAC-2019\\base_data'
DATA_PREFIXS = ["B", "M"]
OUTPUT_PATH = '..\\..\\output\\FNAC'

def splitImage(image_path, output_path):
    # 图片路径，和切割后保存的图片路径
    # 取名为"原名_n".jpg
    
    image = Image.open(image_path)
    width, height = image.size
    MINIMUM = 10000 # 随机选择的数量
    overlap_rate = 0.5 
    crop_size = 256
    overlap = int(crop_size * overlap_rate)

    # 计算水平和垂直方向上的切割数量
    num_horizontal = (width - overlap) // (crop_size - overlap)
    num_vertical = (height - overlap) // (crop_size - overlap)

    cropped_images = []

    for i in range(num_horizontal):
        for j in range(num_vertical):
            left = i * (crop_size - overlap)
            upper = j * (crop_size - overlap)
            right = left + crop_size
            lower = upper + crop_size

            # 切割图像
            crop = image.crop((left, upper, right, lower))
            cropped_images.append(crop)

    # 随机选择要保存的图片
    num_images_to_save = min(MINIMUM, len(cropped_images))
    selected_image_indexs = random.sample(range(len(cropped_images)), num_images_to_save)
    selected_image_indexs = sorted(selected_image_indexs)
    selected_images = [cropped_images[i] for i in selected_image_indexs]

    # 保存切割后的图像
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    for index, img in enumerate(selected_images):
        tmp_output_path = os.path.join(output_path, f"{image_name}_{index}.jpg")
        img.save(tmp_output_path)

def splitImagesInDirectory(path, output_path):
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 遍历目录中的所有文件
    for filename in tqdm(os.listdir(path)):
        # 检查文件是否为图片
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(path, filename)
            splitImage(image_path, output_path)

def main():
    for data_prefix in DATA_PREFIXS:
        input_path = os.path.join(DATA_PATH, data_prefix)
        output_path = os.path.join(OUTPUT_PATH, data_prefix)
        splitImagesInDirectory(input_path, output_path)

if __name__ == "__main__":
    main()