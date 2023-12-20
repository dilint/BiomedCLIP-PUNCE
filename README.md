# biomed-clip-puNCE

## Requirement
```bash
pip install open_clip_torch transformers matplotlib
pip install h5py
pip install scikit-learn
pip install future
```

## Train Backbone
```bash
cd simclr
```

```bash
# 训练backbone（2d 4张3080
python -m torch.distributed.launch --nproc_per_node=4 simclr_ngc.py --ddp --data_dir '/home1/wsi/ngc-2023-1333' --project=$PROJECT_NAME --model_path=$OUTPUT_PATH --title=$TITLE_NAME --seed=2023 --batch_size=64 --epochs=200 --wandb
```

## Extract WSI Features
```bash
cd extract-features
```

```bash
# 提取特征 （20min 8张2080
GPU_NUMBERS=8
FEAT_DIR='resnet_simclr_infonce_ngc_224_4x64_224_trainpart'
CKP_PATH='/root/project/biomed-clip-puNCE/simclr/output-model/resnet_simclr_infonce_ngc_224_4x64_trainpart/resnet_simclr_infonce_ngc_224_4x64_trainpart_epoch130.pt'
python -m torch.distributed.launch --nproc_per_node=$GPU_NUMBERS extract_features_ngc.py --base_model='resnet50' --output_path='result-final-ngc-features' --feat_dir=$FEAT_DIR --ckp_path=$CKP_PATH --target_patch_size 224 224 --multi_gpu
```

## MIL Train
```bash
cd mil-methods
```

```bash
# MIL整合
# abmil  （22min 单张2080
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
TITLE_NAME=resnet3-abmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=0, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=1024 --cv_fold=1 --title=$TITLE_NAME --model=pure --baseline=attn --seed=2023 --wandb

# transmil  （38min 单张2080
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
OUTPUT_PATH=output-model
TITLE_NAME=resnet3-transmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=0, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=1024 --cv_fold=1 --title=$TITLE_NAME --model=pure --baseline=selfattn --seed=2023 --wandb

# meanmil  （13min 单张2080
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
OUTPUT_PATH=output-model
TITLE_NAME=resnet3-meanmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=0, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=1024 --cv_fold=1 --title=$TITLE_NAME --model=meanmil --seed=2023 --wandb

# maxmil （7min 单张2080
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
OUTPUT_PATH=output-model
TITLE_NAME=resnet3-maxmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=0, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --input_dim=1024 --cv_fold=1 --title=$TITLE_NAME --model=maxmil --seed=2023 --wandb

# MHIM(abmil)
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
OUTPUT_PATH=output-model
TITLE_NAME=resnet3-mhim(abmil)-ngc-customsplit
TEACHER_INIT=./output-model/output-test/biomedclip1-abmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=1, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --cv_fold=1 --input_dim=1024 --teacher_init=$TEACHER_INIT --title=$TITLE_NAME --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed=2023 --wandb

# MHIM(transmil)
PROJECT_NAME=output-test
DATASET_PATH=/root/project/clam/result-final-ngc-features/resnet_simclr_infonce_ngc_224_4x64_224_trainpart
OUTPUT_PATH=output-model
TITLE_NAME=resnet3-mhim(transmil)-ngc-customsplit
TEACHER_INIT=./output-model/output-test/biomedclip1-transmil-ngc-customsplit
CUDA_VISIBLE_DEVICES=0, python3 main_custome.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --model_path=$OUTPUT_PATH --datasets=ngc --cv_fold=1 --input_dim=1024 --teacher_init=$TEACHER_INIT --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title=$TITLE_NAME --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed=2023 --wandb

```