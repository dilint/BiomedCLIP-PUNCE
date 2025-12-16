# RTDETR提取特征
## Prepare
1. 获得RTDETR onnx推理模型文件

2. 环境安装
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install h5py opencv-python onnxruntime-gpu
```

## Run
```bash

# 单卡
python extract_features_rtdetr_fast.py
# 多卡
torchrun --nproc_per_node=4 extract_features_rtdetr_fast.py --multi_gpu
```