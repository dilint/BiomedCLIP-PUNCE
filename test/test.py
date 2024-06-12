import open_clip
import torch
from PIL import Image
import os

data_root = '/home/commonfile/TCTAnnotatedData/TCTAnnotated20210331/Annotated202103TCTFK-XIMEA-1200-YX-BZ-1'
wsi_name = '6143430877H--209-G'
input_dir = os.path.join(data_root, wsi_name)
output_dir = 'output'
vis_patch = '2_19'
vis_image = vis_patch + '.jpg'
vis_txt = vis_patch + '.txt'
vis_image_path = os.path.join(input_dir, vis_image)

vis_image_path = '3_31.jpg'
# 打开图片

image = Image.open(vis_image_path)


model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model = model.visual.trunk
model.norm = torch.nn.Identity()
image = preprocess_val(image).unsqueeze(0)
vis_feature = model.forward_features(image)
print(vis_feature.shape)