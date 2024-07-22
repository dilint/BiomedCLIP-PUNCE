import threading
import os
import argparse, os

def run_script(mil, args):
    os.system(f'bash scripts/{mil}mil.sh {args.feature_name} {args.dataset} {args.num_dim}')
    os.system(f'bash scripts/mhim\({mil}mil\).sh {args.feature_name} {args.dataset} {args.num_dim}')
    
parser = argparse.ArgumentParser(description='MIL Training Script')
parser.add_argument('feature_name', default='test', type=str, help='Feature name of extracted patch features')
parser.add_argument('--dataset', default='gc', type=str, help='Dataset name')
parser.add_argument('--num_dim', default=512, type=int, help='number of dimension')

parser.add_argument('--test_mode', action='store_true', help='Test mode')
args = parser.parse_args()

if not args.test_mode:
    # train
    thread1 = threading.Thread(target=run_script, args=('ab', args))
    thread2 = threading.Thread(target=run_script, args=('trans', args))
    
    # 启动线程
    thread1.start()
    thread2.start()

    # 等待所有线程完成
    thread1.join()
    thread2.join()
    
# test
FEATURE_NAME=args.feature_name
DATASET=args.dataset
NUM_DIM=args.num_dim
DATASET_PATH=f'../extract-features/result-final-{DATASET}-features/{FEATURE_NAME}'
LABEL_PATH=f'../datatools/{DATASET}/labels'
OUTPUT_PATH='output-model'
PROJECT_NAME='test'
DEVICE=3080

# abmil
print('abmil')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-abmil-{DATASET}-trainval-{DEVICE}'
os.system(f'CUDA_VISIBLE_DEVICES=0, python3 eval.py --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=attn --seed=2024')
# mhim(abmil)
print('mhim(abmil)')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-mhim\(abmil\)-{DATASET}-trainval-{DEVICE}'
os.system(f'CUDA_VISIBLE_DEVICES=0, python3 eval.py --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=attn --seed=2024')
# transmil
print('transmil')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-transmil-{DATASET}-trainval-{DEVICE}'
os.system(f'CUDA_VISIBLE_DEVICES=0, python3 eval.py --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=selfattn --seed=2024')
# mhim(transmil)
print('mhim(transmil)')
CHECKPOINT_PATH=f'{OUTPUT_PATH}/{PROJECT_NAME}/{FEATURE_NAME}-mhim\(transmil\)-{DATASET}-trainval-{DEVICE}'
os.system(f'CUDA_VISIBLE_DEVICES=0, python3 eval.py --label_path={LABEL_PATH} --dataset_root={DATASET_PATH} --ckp_path={CHECKPOINT_PATH} --datasets=tct --input_dim={NUM_DIM} --model=pure --baseline=selfattn --seed=2024')
