# biomed-clip-puNCE

## 更多实验可见
[飞书文档](https://ci929abihif.feishu.cn/docx/KxBkdxE1FowDrexDpdHcelPCnUd)

## Requirement
```bash
conda create -n biomed python=3.8
pip install jupyter
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install open-clip-torch transformers matplotlib
pip install h5py scikit-learn==0.22.1 future==0.18.3
pip install wandb==0.15 torchsummary==1.5.1, torchmetrics
pip install einops, chardet
```


## Biomed-SimCLR流程
![structure](imgs/structure.jpg)
```bash
# high risk filter
cd ../extract-features
GPU_NUMBERS=4
FEAT_DIR='biomed1'
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=biomedclip --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64

cd ../mil-methods
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=biomed1-meanmil-gc-trainval-test
CUDA_VISIBLE_DEVICES=0, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=512 --cv_fold=1 --title=$TITLE_NAME --model=meanmil --seed=2024 --train_val --wandb

FEATURE_ROOT='../extract-features/result-final-gc-features/biomed1'
WSI_ROOT='/home1/wsi/gc-224'
TRAIN_LABEL='../datatools/tct-gc/labels/train_val.csv'
CKP_PATH='./output-model/mil-methods/biomed1-meanmil-gc/fold_0_model_best_auc.pt'
OUTPUT_ROOT='/home1/wsi/gc-output-filter/meanmil'
python inference-multi.py --input_dim=512 --datasets=gc --feature_root=$FEATURE_ROOT --wsi_root=$WSI_ROOT --output_root=$OUTPUT_ROOT --train_label=$TRAIN_LABEL --ckp_path=$CKP_PATH   --model=meanmil

# train backbone
cd ../extract-features
DATA_DIR='/home1/wsi/gc-output-filter/biomed1-meanmil'
TRAIN_LABEL='../datatools/tct-gc/labels/train_label.csv'
PROJECT_NAME='new-simclr-puc'
OUTPUT_PATH='output-model'
TITLE_NAME='biomed_simclr_infonce_filterGC_224_4*256'
python -m torch.distributed.launch --nproc_per_node=4 simclr.py --ddp --dataset=gc --backbone=biomedCLIP --data_dir=$DATA_DIR --train_label_path=$TRAIN_LABEL --project=$PROJECT_NAME --model_path=$OUTPUT_PATH --title=$TITLE_NAME --seed=2024 --batch_size=256 --epochs=200 --wandb


# extract features
GPU_NUMBERS=4
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
FEAT_DIR='biomed3'
CKP_PATH='output-model/new-simclr-puc/biomed_simclr_infonce_filterGC_224_4*256/biomed_simclr_infonce_filterGC_224_4*256_epoch200.pt'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS --master_port=12345 extract_features_tct.py --base_model=biomedclip --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --with_adapter --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64

# train mil
cd ../mil/scripts
bash abmil.sh biomed3 gc 512
bash transil.sh biomed3 gc 512
bash mhim(abmil).sh biomed3 gc 512
bash transmil(abmil).sh biomed3 gc 512
```


## Train Backbone
```bash
cd simclr
```

```bash
# 训练backbone（4h    4张3080
cd ../extract-features
DATA_DIR='/home1/wsi/gc-output-filter/biomed1-meanmil'
TRAIN_LABEL='../datatools/tct-gc/labels/train_label.csv'
PROJECT_NAME='new-simclr-puc'
OUTPUT_PATH='output-model'
TITLE_NAME='biomed_simclr_infonce_filterGC_224_4*256'
python -m torch.distributed.launch --nproc_per_node=4 simclr.py --ddp --dataset=gc --backbone=biomedCLIP --data_dir=$DATA_DIR --train_label_path=$TRAIN_LABEL --project=$PROJECT_NAME --model_path=$OUTPUT_PATH --title=$TITLE_NAME --seed=2024 --batch_size=256 --epochs=200 --wandb
```



## Extract WSI Features
```bash
cd extract-features
```

```bash
# 使用biomedCLIP提取特征
GPU_NUMBERS=4
FEAT_DIR='biomed1'
WSI_ROOT='/home1/wsi/gc-224'
OUTPUT_PATH='result-final-gc-features'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_tct.py --base_model=biomedclip --dataset=gc --output_path=$OUTPUT_PATH --feat_dir=$FEAT_DIR --wsi_root=$WSI_ROOT --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu --batch_size=32 --num_workers=64
```

## MIL Train
```bash
cd mil-methods
```

```bash
# MIL整合 
# meanmil
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=biomed3-maxmil-gc-trainval-test
CUDA_VISIBLE_DEVICES=0, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=512 --cv_fold=1 --title=$TITLE_NAME --model=meanmil --seed=2024 --train_val --wandb

# maxmil
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=biomed2-maxmil-tct-trainval
CUDA_VISIBLE_DEVICES=0, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=768 --cv_fold=1 --title=$TITLE_NAME --model=maxmil --seed=2024 --train_val --wandb

# abmil 
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=biomed2-abmil-tct-trainval
CUDA_VISIBLE_DEVICES=1, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=768 --cv_fold=1 --title=$TITLE_NAME --model=pure --baseline=attn --train_val --seed=2024 --wandb

# transmil 
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME=biomed2-transmil-tct-trainval
CUDA_VISIBLE_DEVICES=2, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=768 --cv_fold=1 --title=$TITLE_NAME --model=pure --train_val --baseline=selfattn --seed=2024 --wandb

# MHIM(abmil)
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME='biomed1-mhim(abmil)-tct-trainval'
TEACHER_INIT=./output-model/mil-methods/biomed1-abmil-tct-trainval
CUDA_VISIBLE_DEVICES=0, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --label_path=$LABEL_PATH --datasets=ngc --cv_fold=1 --input_dim=768 --teacher_init=$TEACHER_INIT --title=$TITLE_NAME --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2024 --wandb

# MHIM(transmil)
DATASET_PATH=../extract-features/result-final-gc-features/biomed1
LABEL_PATH=../datatools/tct-gc/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods
TITLE_NAME='biomed1-mhim(transmil)-tct-trainval'
TEACHER_INIT=./output-model/mil-methods/biomed1-transmil-tct-trainval
CUDA_VISIBLE_DEVICES=0, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH  --label_path=$LABEL_PATH --datasets=ngc --cv_fold=1 --input_dim=768 --teacher_init=$TEACHER_INIT --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title=$TITLE_NAME --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2024 --wandb
```

