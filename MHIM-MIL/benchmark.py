import threading
import os
import argparse, os

def run_script(mil):
    if mil == 'ab':
        TITLE_NAME=f'{FEATURE_NAME}-abmil-{DATASET}-trainval-{SERVER}-{SEED}'
        os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 mil.py --n_classes=5 --project={PROJECT_NAME} --dataset_root={DATASET_PATH} --label_path={LABEL_PATH} --model_path={OUTPUT_PATH} --datasets={DATASET} --input_dim={NUM_DIM} --cv_fold=1 --title={TITLE_NAME} --model=pure --baseline=attn --train_val --seed={SEED} --wandb')

        TITLE_NAME=f'{FEATURE_NAME}-mhim\(abmil\)-{DATASET}-trainval-{SERVER}-{SEED}'
        TEACHER_INIT=f'./{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-abmil-{DATASET}-trainval-{SERVER}'
        # os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 mil.py --n_classes=5 --project={PROJECT_NAME} --dataset_root={DATASET_PATH} --model_path={OUTPUT_PATH} --label_path={LABEL_PATH} --datasets={DATASET} --cv_fold=1 --input_dim={NUM_DIM} --train_val --teacher_init={TEACHER_INIT} --title={TITLE_NAME} --model=mhim --baseline=attn --num_workers=0 --cl_alpha=0.1 --mask_ratio_h=0.01 --mask_ratio_hr=0.5 --mrh_sche --init_stu_type=fc --mask_ratio=0.5 --mask_ratio_l=0. --seed={SEED} --wandb')
    elif mil == 'trans':
        TITLE_NAME=f'{FEATURE_NAME}-transmil-{DATASET}-trainval-{SERVER}-{SEED}'
        os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID2}, python3 mil.py --n_classes=5 --project={PROJECT_NAME} --dataset_root={DATASET_PATH} --label_path={LABEL_PATH} --model_path={OUTPUT_PATH} --datasets={DATASET} --input_dim={NUM_DIM} --cv_fold=1 --title={TITLE_NAME} --model=pure --baseline=selfattn --train_val --seed={SEED} --wandb')

        TITLE_NAME=f'{FEATURE_NAME}-mhim\(transmil\)-{DATASET}-trainval-{SERVER}-{SEED}'
        TEACHER_INIT=f'./{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-transmil-{DATASET}-trainval-{SERVER}'
        # os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID2}, python3 mil.py --n_classes=5 --project={PROJECT_NAME} --dataset_root={DATASET_PATH} --model_path={OUTPUT_PATH} --label_path={LABEL_PATH} --datasets={DATASET} --cv_fold=1 --input_dim={NUM_DIM} --train_val --teacher_init={TEACHER_INIT} --model=mhim --baseline=selfattn --mask_ratio_h=0.03 --mask_ratio_hr=0.5 --mrh_sche --title={TITLE_NAME} --mask_ratio=0. --mask_ratio_l=0.8 --cl_alpha=0.1 --mm_sche --init_stu_type=fc --attn_layer=0 --seed={SEED} --wandb')
    elif mil == 'clam':
        TITLE_NAME=f'{FEATURE_NAME}-clam-{DATASET}-trainval-{SERVER}-{SEED}'
        os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 mil.py --n_classes=5 --project={PROJECT_NAME} --dataset_root={DATASET_PATH} --label_path={LABEL_PATH} --model_path={OUTPUT_PATH} --datasets={DATASET} --input_dim={NUM_DIM} --cv_fold=1 --title={TITLE_NAME} --model=clam_sb --train_val --seed={SEED} --wandb')

parser = argparse.ArgumentParser(description='MIL Training Script')
parser.add_argument('feature_name', default='test', type=str, help='Feature name of extracted patch features')
parser.add_argument('--dataset', default='gc', type=str, help='Dataset name')
parser.add_argument('--num_dim', default=512, type=int, help='number of dimension')
parser.add_argument('--seed', default=2024, type=int, help='seed')
parser.add_argument('--cpus', type=int, nargs='+', default=(0, 1))

parser.add_argument('--test_mode', action='store_true' ,help='Test mode')
args = parser.parse_args()

FEATURE_NAME=args.feature_name
DATASET=args.dataset
NUM_DIM=args.num_dim
SEED=args.seed
DATASET_PATH=f'/home1/wsi/gc-all-features/frozen/{FEATURE_NAME}'
LABEL_PATH=f'../datatools/{DATASET}/n-labels'
OUTPUT_PATH='output-model'
PROJECT_NAME='test'
SERVER=3080
CPU_ID1, CPU_ID2 = args.cpus

if not args.test_mode:
    # train
    thread1 = threading.Thread(target=run_script, args=('ab',))
    thread2 = threading.Thread(target=run_script, args=('trans',))
    thread3 = threading.Thread(target=run_script, args=('clam',))
    
    # 启动线程
    thread1.start()
    thread2.start()
    thread3.start()
    
    # 等待所有线程完成
    thread1.join()
    thread2.join()
    thread3.join()
    
# test

# abmil
print('==================================\033[35mabmil\033[0m==================================')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-abmil-{DATASET}-trainval-{SERVER}-{SEED}'
os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 eval.py --n_classes=5 --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=attn --seed={SEED}')
# mhim(abmil)
# print('==================================\033[35mmhim(abmil)\033[0m==================================')
# CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-mhim\(abmil\)-{DATASET}-trainval-{SERVER}-{SEED}'
# os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 eval.py --n_classes=5 --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=attn --seed={SEED}')
# transmil
print('==================================\033[35mtransmil\033[0m==================================')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-transmil-{DATASET}-trainval-{SERVER}-{SEED}'
os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 eval.py --n_classes=5 --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=selfattn --seed={SEED}')
# mhim(transmil)
# print('==================================\033[35mmhim(transmil)\033[0m==================================')
# CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-mhim\(transmil\)-{DATASET}-trainval-{SERVER}-{SEED}'
# os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 eval.py --n_classes=5 --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=selfattn --seed={SEED}')
# clam
print('==================================\033[35mclam\033[0m==================================')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-clam-{DATASET}-trainval-{SERVER}-{SEED}'
os.system(f'CUDA_VISIBLE_DEVICES={CPU_ID1}, python3 eval.py --n_classes=5 --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=clam_sb --seed={SEED}')
