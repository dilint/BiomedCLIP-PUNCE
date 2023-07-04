
import open_clip
import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATA_PATH = '..\\output\\FNAC'
DATA_PREFIXS = ["B", "M"]
OUTPUT_PATH = '..\\output\\FNAC-features'
TEMPLATE = 'this is a photo of '
BATCH_SIZE = 256
labels = [
    'adenocarcinoma histopathology',    # 阴性标签
    'squamous cell carcinoma histopathology',   # 阳性标签
]

# 自定义数据集
class CustomDataset(Dataset):
    def __init__(self, root, transform=None, device=None):
        self.root = root
        self.transform = transform
        self.device = device
        self.file_list = os.listdir(root)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.file_list[index])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.device is not None:
            img = img.to(self.device)
        label = 0  # 默认所有图像的类别为0
        wsi_num = os.path.basename(img_path).split("_")[0]
        return img, label, wsi_num
 

def _info(mes: str):
    print("------------Info:" + mes)
    
def main():
    # 载入模型文件
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    _info("Biomed CLIP模型载入完成")
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    context_length = 256
    texts = tokenizer([TEMPLATE + l for l in labels], context_length=context_length).to(device)

    # ---function start---
    # 从input_path读入图片转化为pth保存到output_path
    # 通过n_m.jpg的n判断图片属于哪一张wsi
    def extractFeaturesFromOneDirectories(input_path: str, output_path: str):   
        # 创建 CustomDataset 数据集
        # dataset = CustomDataset(root='D:/dataset/FNAC-2019/B', transform=preprocess_val, device=device)
        dataset = CustomDataset(root=input_path, transform=preprocess_val, device=device)
        # 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        model.to(device)
        model.eval()
        scores = {}
        all_image_features = {}

        with torch.no_grad():
            # 提取特征
            _info(f"开始提取{os.path.abspath(input_path)}下的图片特征")
            for images, _, wsi_nums in tqdm(dataloader):
                # image_features @ text_features.t()执行了图像特征和文本特征之间的点积操作，生成了一个分数矩阵。然后，通过乘以logit_scale对分数矩阵进行缩放。这个缩放过程可以增加或减小点积的值，从而影响最终的logits结果。
                image_features, text_features, logit_scale = model(images, texts)
                logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
                logits = logits.cpu().numpy()
                for logit, image_feature, wsi_num in zip(logits, image_features, wsi_nums): 
                    if wsi_num not in scores:
                        scores[wsi_num] = []
                    if wsi_num not in all_image_features:
                        all_image_features[wsi_num] = []
                    scores[wsi_num].append(logit[1])
                    all_image_features[wsi_num].append(image_feature)
                    
        # 存储pth
        _info(f"开始保存{os.path.abspath(input_path)}下的图片特征")
        for wsi_num in tqdm(scores.keys()):
            tmp_image_features = all_image_features[wsi_num]
            tmp_scores = scores[wsi_num]

            # 使用 enumerate 函数获取 all_image_features 的索引和值，并构建排序键
            sorted_data = sorted(enumerate(tmp_scores), key=lambda x: x[1], reverse=True)

            # 选择前 100 个 all_image_features 的索引
            selected_indices = [data[0] for data in sorted_data[:100]]
            # 保持原本顺序
            selected_indices.sort()
            # 根据索引提取对应的 all_image_features
            selected_features = [tmp_image_features[idx] for idx in selected_indices]

            # 保存 selected_features
            torch.save(selected_features, os.path.join(output_path, wsi_num + '.pth'))             
    # ---function end---

    for data_prefix in DATA_PREFIXS:
        input_path = os.path.join(DATA_PATH, data_prefix)
        output_path = os.path.join(OUTPUT_PATH, data_prefix)
        if not os.path.exists(output_path): 
            try:
                os.makedirs(output_path)
            except OSError:
                print(f"Failed to create folder '{output_path}'.")

        extractFeaturesFromOneDirectories(input_path, output_path)
    
if __name__ == "__main__":
    main()