import timm
from PIL import Image
from torchvision import transforms
import torch

tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
tile_encoder = tile_encoder.to('cuda')

transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

img_path = "3_31.jpg"
sample_input = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to('cuda')

tile_encoder.eval()
with torch.no_grad():
    output = tile_encoder(sample_input).squeeze()
    print(output.shape)