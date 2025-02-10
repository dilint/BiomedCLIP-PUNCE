import torch
from torchvision import datasets, transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT 通常需要 224x224 的输入
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

import timm
from open_clip.timm_model import TimmModel
from torch import nn

# 加载预训练的 ViT 模型
model_name = "vit_base_patch16_224"  # 选择 ViT 模型
model = timm.create_model(model_name, pretrained=True, num_classes=10)  # 修改分类头为 10 类
timm_model_name = 'vit_base_patch16_224'
timm_model_pretrained = False
timm_pool = ""
timm_proj = 'linear'
image_size = 224
embed_dim=512
model = TimmModel(
    timm_model_name,
    embed_dim,
    pretrained=timm_model_pretrained,
    pool=timm_pool,
    proj=timm_proj,
    image_size=image_size,
)
model = nn.Sequential(
    model,
    nn.Linear(512, 10)  # 添加一个线性层用于分类
)
# 打印模型结构
# print(model)
import torch.nn as nn
import torch.optim as optim

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 将模型移动到 GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练循环
for epoch in range(10):  # 训练 10 个 epoch
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 打印一次损失
            print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100}')
            running_loss = 0.0
            
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on test set: {100 * correct / total}%')