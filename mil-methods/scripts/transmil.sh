echo "提取特征方式 ：$1"
echo "数据集 ：$2"
echo "输入维度 ：$3"
echo "高风险权重 ：$4"
GPU_ID=3
SERVER=2080

FEATURE_NAME=$1
DATASET=$2
NUM_DIM=$3
HIGH_WEIGHT=$4

cd ../
DATASET_PATH=../extract-features/result-final-$DATASET-features/$FEATURE_NAME
LABEL_PATH=../datatools/$DATASET/labels
OUTPUT_PATH=output-model
PROJECT_NAME=mil-methods-info
# TITLE_NAME=$FEATURE_NAME-transmil-$DATASET-trainval-$HIGH_WEIGHT
TITLE_NAME=$FEATURE_NAME-transmil-$DATASET-trainval-$SERVER
# CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=$DATASET --input_dim=$NUM_DIM --cv_fold=1 --title=$TITLE_NAME --model=pure --train_val --baseline=selfattn --seed=2024 --high_weight=$HIGH_WEIGHT --wandb
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 mil.py --project=$PROJECT_NAME --dataset_root=$DATASET_PATH --label_path=$LABEL_PATH --model_path=$OUTPUT_PATH --datasets=$DATASET --input_dim=$NUM_DIM --cv_fold=1 --title=$TITLE_NAME --model=pure --train_val --baseline=selfattn --seed=2024 --wandb

CHECKPOINT_PATH=$OUTPUT_PATH/$PROJECT_NAME/$FEATURE_NAME-transmil-$DATASET-trainval-$SERVER
CUDA_VISIBLE_DEVICES=$GPU_ID, python3 eval.py --label_path=$LABEL_PATH  --dataset_root=$DATASET_PATH --ckp_path=$CHECKPOINT_PATH --datasets=tct --input_dim=$NUM_DIM --model=pure --baseline=selfattn --seed=2024 