import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score,recall_score, roc_curve, precision_recall_curve
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from prettytable import PrettyTable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import torch
import os


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False   

@torch.no_grad()
def ema_update(model,targ_model,mm=0.9999):
    r"""Performs a momentum update of the target network's weights.
    Args:
        mm (float): Momentum used in moving average update.
    """
    assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm

    for param_q, param_k in zip(model.parameters(), targ_model.parameters()):
        param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm) # mm*k +(1-mm)*q

def patch_shuffle(x,group=0,g_idx=None,return_g_idx=False):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))

    # padding
    H, W = int(np.ceil(np.sqrt(p))), int(np.ceil(np.sqrt(p)))
    if group > H or group<= 0:
        return group_shuffle(x,group)
    _n = -H % group
    H, W = H+_n, W+_n
    add_length = H * W - p
    # print(add_length)
    ps = torch.cat([ps,torch.tensor([-1 for i in range(add_length)])])
    # patchify
    ps = ps.reshape(shape=(group,H//group,group,W//group))
    ps = torch.einsum('hpwq->hwpq',ps)
    ps = ps.reshape(shape=(group**2,H//group,W//group))
    # shuffle
    if g_idx is None:
        g_idx = torch.randperm(ps.size(0))
    ps = ps[g_idx]
    # unpatchify
    ps = ps.reshape(shape=(group,group,H//group,W//group))
    ps = torch.einsum('hwpq->hpwq',ps)
    ps = ps.reshape(shape=(H,W))
    idx = ps[ps>=0].view(p)
    
    if return_g_idx:
        return x[:,idx.long()],g_idx
    else:
        return x[:,idx.long()]

def group_shuffle(x,group=0):
    b,p,n = x.size()
    ps = torch.tensor(list(range(p)))
    if group > 0 and group < p:
        _pad = -p % group
        ps = torch.cat([ps,torch.tensor([-1 for i in range(_pad)])])
        ps = ps.view(group,-1)
        g_idx = torch.randperm(ps.size(0))
        ps = ps[g_idx]
        idx = ps[ps>=0].view(p)
    else:
        idx = torch.randperm(p)
    return x[:,idx.long()]


def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))
    labels = np.array(dataset.slide_label)
    label_uni = set(dataset.slide_label)
    weight_per_class = [N/len(labels[labels==c]) for c in label_uni]
    weight = [0] * int(N)
    for idx in range(len(dataset)):
        y = dataset.slide_label[idx]
        weight[idx] = weight_per_class[y]

    return torch.DoubleTensor(weight)

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], loss[idx], thresholds[idx]

def evaluation_cancer_sigmoid(gt_logtis, pred_logits, class_labels, wsi_names, threshold):
    """
    参数：
        gt_logtis (list): [N, num_class], 真实标签
        pred_logits (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'BV', 'M', 'T'],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    TODO 目前对于多类别任务，只考虑了1,5,3的多分类划分方式以及5分类的单任务模式
    """
    # 对于多类别样本 拆分成多个样本，预测概率将正确的其他类别概率设为0
    
    id2labelcode = {
        0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        2: [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        4: [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    
    assert len(class_labels) == 6 
    bag_labels = []
    new_pred_logits = []
    
    for i, gt_logit in enumerate(gt_logtis):
        gt_labels = torch.where(gt_logit == 1)[0]
        if len(gt_labels) > 1:
            assert f"ground truth multi labels"
        else:
            bag_labels.append(gt_labels[0])
            new_pred_logits.append(pred_logits[i])
            
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(new_pred_logits)
    
    
    bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
 
    roc_auc = []
    thresholds = []
    # 首先评估宫颈癌症风险
    n_cancer_class = len(class_labels_cancer)
    bag_labels_cancer_onehot = np.array([id2labelcode[l] for l in bag_labels_cancer])
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)
    for i in range(1, n_cancer_class):
        # 计算auc和threshold
        fpr, tpr, threshold_ = roc_curve(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i], pos_label=1)
        precision_pr, recall_pr, thresholds_pr = precision_recall_curve(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i])
        f1_scores_pr = 2 * (precision_pr * recall_pr) / (precision_pr + recall_pr + 1e-10)
        best_idx = np.argmax(f1_scores_pr)
        threshold_optimal = thresholds_pr[best_idx]
        best_f1 = f1_scores_pr[best_idx]
        # fpr_optimal, tpr_optimal, socre, threshold_optimal = optimal_thresh(fpr, tpr, threshold_)
        print(i, best_idx, best_f1, threshold_optimal)
        thresholds.append(threshold_optimal)
        roc_auc.append(roc_auc_score(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i]))
        bag_pred_cancer_onehot[:, i] = bag_logits_cancer[:, i] >= threshold_optimal
    # print(bag_pred_cancer_onehot.shape)
    for j in range(bag_pred_cancer_onehot.shape[0]):
        if np.sum(bag_pred_cancer_onehot[j]) == 0:
            bag_pred_cancer_onehot[j, 0] = 1
        elif np.sum(bag_pred_cancer_onehot[j]) > 1:
            # 多个类别都大于阈值，则保留得分最高的类别
            # bag_pred_cancer_onehot[j] = 0
            # bag_pred_cancer_onehot[j, np.argmax(bag_logits_cancer[j, 1:])+1] = 1
            # 多个类别都大于阈值，则保留评级类别高的
            rank=[0,2,1,3,4]
            indices = np.where(bag_pred_cancer_onehot[j, 1:] == 1)[0]
            bag_pred_cancer_onehot[j] = 0
            if len(indices) > 0:
                # 根据 rank 值选择优先级最高的索引
                selected_index = max(indices, key=lambda x: rank[x])
                bag_pred_cancer_onehot[j, selected_index+1] = 1
            
            # 如果该类别被判定为NILM的概率过高 也输出错误信息
            if bag_logits_cancer[j, 0] > 0.95:
                print(f'[ERROR] {wsi_names[j]} risk prediction is wrong: {[round(risk, 4) for risk in bag_logits_cancer[j]]}')
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracy = accuracy_score(bag_labels_cancer, bag_pred_cancer)
    accuracys = [accuracy]
    recalls = recall_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    precisions = precision_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    fscores = f1_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    print('[INFO] confusion matrix for cancer labels:')
    cancer_matrix = confusion_matrix(bag_labels_cancer, bag_pred_cancer, class_labels_cancer)
    print('fscores len' + str(len(fscores)))
    
    # 评估微生物感染
    microbial_matrix = None
    print('Recalls: ' + str(recalls))
    print('roc', 'acc', 'recall', 'prec', 'fs')
    print(roc_auc, accuracys, recalls, precisions, fscores)
    return roc_auc, accuracys, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix
    

def evaluation_cancer_sigmoid_cascade(gt_logtis, pred_logits, class_labels, output_path):
    """
    参数：
        gt_logtis (list): [N, num_class], 真实标签
        pred_logits (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'BV', 'M', 'T'],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    """
    # 对于多类别样本 拆分成多个样本，预测概率将正确的其他类别概率设为0
    
    id2labelcode = {
        0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        2: [0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        3: [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        4: [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    
    assert len(class_labels) == 6 
    bag_labels = []
    new_pred_logits = []
    
    for i, gt_logit in enumerate(gt_logtis):
        gt_labels = torch.where(gt_logit == 1)[0]
        if len(gt_labels) > 1:
            assert f"ground truth multi labels"
        else:
            bag_labels.append(gt_labels[0])
            new_pred_logits.append(pred_logits[i])
            
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(new_pred_logits)
    
    bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
 
    roc_auc = []
    thresholds = []
    n_cancer_class = len(class_labels_cancer)
    bag_labels_cancer_onehot = np.array([id2labelcode[l] for l in bag_labels_cancer])
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)
    # 计算f1最大阈值
    for i in range(1, n_cancer_class):
        # 计算auc和threshold
        precision_pr, recall_pr, thresholds_pr = precision_recall_curve(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i])
        f1_scores_pr = 2 * (precision_pr * recall_pr) / (precision_pr + recall_pr + 1e-10)
        best_idx = np.argmax(f1_scores_pr)
        threshold_optimal = thresholds_pr[best_idx]
        best_f1 = f1_scores_pr[best_idx]
        # fpr_optimal, tpr_optimal, socre, threshold_optimal = optimal_thresh(fpr, tpr, threshold_)
        print(i, best_idx, best_f1, threshold_optimal)
        thresholds.append(threshold_optimal)
        roc_auc.append(roc_auc_score(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i]))
        bag_pred_cancer_onehot[:, i] = bag_logits_cancer[:, i] >= threshold_optimal
    # 进行类别修正：1. 多个阳性类别采用风险最高类别，2. 阴性得分不够高采用logit最大类别
    threshold_neg = 0.98
    for j in range(bag_pred_cancer_onehot.shape[0]):
        if np.sum(bag_pred_cancer_onehot[j]) > 1:
            rank=[0,2,1,3,4]
            indices = np.where(bag_pred_cancer_onehot[j, 1:] == 1)[0]
            bag_pred_cancer_onehot[j] = 0
            if len(indices) > 0:
                # 根据 rank 值选择优先级最高的索引
                selected_index = max(indices, key=lambda x: rank[x])
                bag_pred_cancer_onehot[j, selected_index+1] = 1
        elif np.sum(bag_pred_cancer_onehot[j]) == 0:
            if bag_logits_cancer[j, 0] > threshold_neg:
                bag_pred_cancer_onehot[j, 0] = 1
            else:
                max_logit_index = np.argmax(bag_logits_cancer[j,1:]) 
                bag_pred_cancer_onehot[j, max_logit_index+1] = 1 
            
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracys, recalls, precisions, fscores = cal_evaluation(bag_labels_cancer, bag_pred_cancer)
    print('[INFO] confusion matrix for cancer labels:')
    cancer_matrix = confusion_matrix(bag_labels_cancer, bag_pred_cancer, class_labels_cancer)

    gt_binary = np.where(bag_labels_cancer != 0, 1, 0)
    pred_binary = np.where(bag_pred_cancer != 0, 1, 0)
    accuracys_bi, recalls_bi, precisions_bi, fscores_bi = cal_evaluation(gt_binary, pred_binary)
    cancer_matrix_bi = confusion_matrix(gt_binary, pred_binary, ['neg', 'pos'])
    roc_bi = roc_auc_score(1-gt_binary, bag_logits_cancer[:, 0])
    
    try:
        with pd.ExcelWriter(output_path) as writer:
            save_metrics_to_excel(roc_auc, accuracys, recalls, precisions, fscores, thresholds, cancer_matrix, class_labels, writer)
            save_metrics_to_excel([roc_bi], accuracys_bi, recalls_bi, precisions_bi, fscores_bi, [threshold_neg], cancer_matrix_bi, ['neg', 'pos'], writer)
        print(f"Metrics and confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving to Excel file {output_path}: {e}")
    
    print('roc', 'acc', 'recall', 'prec', 'fs')
    print(roc_auc, accuracys, recalls, precisions, fscores)
    return roc_auc, accuracys, recalls, precisions, fscores, thresholds, \
        accuracys_bi, recalls_bi, precisions_bi, fscores_bi, \
        cancer_matrix, cancer_matrix_bi


def evaluation_cancer_sigmoid_cascade_binary(gt_logtis, pred_logits, class_labels, output_path):
    """
    参数：
        gt_logtis (list): [N, num_class], 真实标签
        pred_logits (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'BV', 'M', 'T'],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    """
    # 对于多类别样本 拆分成多个样本，预测概率将正确的其他类别概率设为0
    
    bag_labels = []
    new_pred_logits = []
    
    for i, gt_logit in enumerate(gt_logtis):
        gt_labels = torch.where(gt_logit == 1)[0]
        if len(gt_labels) > 1:
            assert f"ground truth multi labels"
        else:
            bag_labels.append(gt_labels[0])
            new_pred_logits.append(pred_logits[i])
            
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(new_pred_logits)
    
    bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)

    threshold_neg = 0.5
    for j in range(bag_logits_cancer.shape[0]):
        if bag_logits_cancer[j, 0] > threshold_neg:
            bag_pred_cancer_onehot[j, 0] = 1
        else:
            max_logit_index = np.argmax(bag_logits_cancer[j,1:]) 
            bag_pred_cancer_onehot[j, max_logit_index+1] = 1
            
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    gt_binary = np.where(bag_labels_cancer != 0, 1, 0)
    pred_binary = np.where(bag_pred_cancer != 0, 1, 0)
    accuracys_bi, recalls_bi, precisions_bi, fscores_bi = cal_evaluation(gt_binary, pred_binary)
    cancer_matrix_bi = confusion_matrix(gt_binary, pred_binary, ['neg', 'pos'])
    roc_bi = roc_auc_score(gt_binary, 1-bag_logits_cancer[:, 0])
    
    try:
        with pd.ExcelWriter(output_path) as writer:
            save_metrics_to_excel([roc_bi], accuracys_bi, recalls_bi, precisions_bi, fscores_bi, [threshold_neg], cancer_matrix_bi, ['neg', 'pos'], writer)
        print(f"Metrics and confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving to Excel file {output_path}: {e}")
    
    print('roc', 'acc', 'recall', 'prec', 'fs')
    print(roc_bi, accuracys_bi, recalls_bi, precisions_bi, fscores_bi)
    return None, None, None, None, None, None, \
        accuracys_bi, recalls_bi, precisions_bi, fscores_bi, \
        None, cancer_matrix_bi
    
def evaluation_cancer_softmax(gt_logtis, pred_logits, class_labels, output_path):
    """
    参数：
        bag_labels (list): [N], 真实标签
        pred_logits (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC', 'BV', 'M', 'T'],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    TODO 目前对于多类别任务，只考虑了1,5,3的多分类划分方式以及5分类的单任务模式
    """
    # 对于多类别样本 拆分成多个样本，预测概率将正确的其他类别概率设为0

    assert len(class_labels) == 6 or len(class_labels) == 5 or len(class_labels) == 2
    gt_logtis = np.array(gt_logtis)
    bag_logits = np.array(pred_logits)
    bag_labels = np.argmax(gt_logtis, axis=1)
    pred_labels = np.argmax(bag_logits, axis=1)
    n_cancer_class = bag_logits.shape[1]
    print(gt_logtis.shape)
    print(bag_logits.shape)
    roc_auc, thresholds = [], []
    for i in range(1, n_cancer_class):
        roc_auc.append(roc_auc_score(gt_logtis[:, i], bag_logits[:, i]))
        thresholds.append(0)
    # bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracys, recalls, precisions, fscores = cal_evaluation(bag_labels, pred_labels)
    print('[INFO] confusion matrix for cancer labels:')
    cancer_matrix = confusion_matrix(bag_labels, pred_labels, class_labels)
    # 评估微生物感染
    microbial_matrix = None
    try:
        with pd.ExcelWriter(output_path) as writer:
            save_metrics_to_excel(roc_auc, accuracys, recalls, precisions, fscores, [0], cancer_matrix, ['neg', 'pos'], writer)
        print(f"Metrics and confusion matrix saved to {output_path}")
    except Exception as e:
        print(f"Error saving to Excel file {output_path}: {e}")
    return roc_auc, accuracys, recalls, precisions, fscores, thresholds, cancer_matrix, microbial_matrix


def cal_evaluation(bag_labels, pred_labels):
    n_classes = max(bag_labels)+1
    accuracy = accuracy_score(bag_labels, pred_labels)
    recalls = recall_score(bag_labels, pred_labels, average=None, labels=list(range(1,n_classes)))
    precisions = precision_score(bag_labels, pred_labels, average=None, labels=list(range(1,n_classes)))
    fscores = f1_score(bag_labels, pred_labels, average=None, labels=list(range(1,n_classes)))
    return [accuracy], recalls, precisions, fscores
    
def confusion_matrix(bag_labels, bag_pred, class_labels):
    """
    混淆矩阵生成：
    参数：
        bag_labels (ndarray): [N] 真实标签
        bag_pred (ndarray): [N] 预测标签
        class_labels (list): n_class 标签名称
    """
    if len(class_labels) == 2:
        y_true, y_pred = [1 if i != 0 else 0 for i in bag_labels], [1 if i != 0 else 0 for i in bag_pred]
        # if isinstance(bag_logits[0], np.ndarray):
        #     y_true, y_pred = bag_labels, np.argmax(np.array(bag_logits), axis=-1)
        # else:
        #     y_true, y_pred = bag_labels, np.array([1 if x > 0.5 else 0 for x in bag_logits])
            
    y_true, y_pred = bag_labels, bag_pred
    num_classes = len(class_labels)
    print(max(y_true), max(y_pred), num_classes)

    # 初始化混淆矩阵
    cm_manual = np.zeros((num_classes, num_classes), dtype=int)

    # 遍历数据，填充混淆矩阵
    for true, pred in zip(y_true, y_pred):
        cm_manual[true][pred] += 1

    row_totals = [sum(row) for row in cm_manual]
    col_totals = [sum(col) for col in zip(*cm_manual)]
    total = sum(row_totals)

    # 重新格式化混淆矩阵，确保第一行包含类别名称
    print(f"Confusion Matrix for {len(bag_labels)} data")
    table = PrettyTable()
    table.field_names = ["实际\预测"] + class_labels + ["总计"]
    for i, label in enumerate(class_labels):
        table.add_row([label] + list(map(str, cm_manual[i])) + [row_totals[i]])
    table.add_row(["总计"] + list(map(str, col_totals)) + [total])
    print(table)
    return table

def prettytable_to_dataframe(pt):
    """
    将 PrettyTable 转换为 pandas DataFrame。
    
    参数：
        pt (PrettyTable): PrettyTable 对象。
    返回：
        pd.DataFrame: 转换后的 DataFrame。
    """
    # 获取表头和行数据
    headers = pt.field_names
    rows = pt._rows

    # 转换为 DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

from typing import List, Union, Any

def save_metrics_to_excel(roc_auc: List[float], accuracies: List[float], recalls: List[float], 
                          precisions: List[float], fscores: List[float], thresholds: List[float], 
                          confusion_matrix: Any, class_labels: List[str], writer):
    """
    将指标存储到Excel表格中，兼容多分类（NILM vs 多个阳性类）和二分类（阴性 vs 阳性）。
    
    参数：
        roc_auc (list): 每个阳性类别的AUC值。
        accuracies (list): 包含多分类（或二分类）准确率。
        recalls (list): 每个阳性类别的召回率。
        precisions (list): 每个阳性类别的精确率。
        fscores (list): 每个阳性类别的F1分数。
        thresholds (list): 每个阳性类别的最优阈值。
        confusion_matrix (Any): 混淆矩阵对象（PrettyTable/numpy array 格式）。
        class_labels (list): 类别标签，例如 ['NILM', 'ASC-US', ...] 或 ['Negative', 'Positive']。
        output_excel_path (str): 输出Excel文件的路径。
    """
    # 判断是否为二分类：class_labels长度为2，或阳性指标长度为1
    is_binary = len(class_labels) == 2 or len(roc_auc) == 1
    
    # 转换为百分比并保留两位小数
    roc_auc_p = [round(auc * 100, 2) for auc in roc_auc]
    recalls_p = [round(recall * 100, 2) for recall in recalls]
    precisions_p = [round(precision * 100, 2) for precision in precisions]
    fscores_p = [round(fscore * 100, 2) for fscore in fscores]
    thresholds_p = [round(threshold, 2) for threshold in thresholds]
    accuracies_p = [round(acc * 100, 2) for acc in accuracies]

    # ----------------------------------------------------
    # 2. 类别指标 DataFrame
    # ----------------------------------------------------
    # 阳性类别的名称 (Class 1 到 N)
    class_names = class_labels[1:]
    results = {
        "Class": class_names,
        "AUC (%)": roc_auc_p,
        "Recall (%)": recalls_p,
        "Precision (%)": precisions_p,
        "F1 Score (%)": fscores_p,
        "Accuracy (%)": ['-'] * len(roc_auc_p),
        "Threshold": thresholds_p,
    }
    df = pd.DataFrame(results)

    # ----------------------------------------------------
    # 3. 平均/总体指标计算与整合
    # ----------------------------------------------------
    avg_data = []
    
    # 主任务总体指标 (Macro-average over all positive classes)
    if roc_auc_p:
        avg_auc = round(np.mean(roc_auc_p), 2)
        avg_recall = round(np.mean(recalls_p), 2)
        avg_precision = round(np.mean(precisions_p), 2)
        avg_fscore = round(np.mean(fscores_p), 2)
        # 假设 accuracies[0] 是总准确率
        total_accuracy = accuracies_p[0] if accuracies_p else '-' 
        # 根据模式调整行名
        row_name = "Binary (Pos vs Neg)" if is_binary else "Macro Average (Pos Classes)"
        
        avg_data.append({
            "Class": row_name,
            "AUC (%)": avg_auc,
            "Recall (%)": avg_recall,
            "Precision (%)": avg_precision,
            "F1 Score (%)": avg_fscore,
            "Accuracy (%)": total_accuracy,
            "Threshold": '-'
        })

    df_avg = pd.DataFrame(avg_data)
    df_final = pd.concat([df, df_avg], ignore_index=True)

    # ----------------------------------------------------
    # 4. 混淆矩阵转换与保存
    # ----------------------------------------------------
    confusion_matrix_df = prettytable_to_dataframe(confusion_matrix)
    metrics_name = "Multi-class Metrics"
    matrix_name = "Multi-class Confusion Matrix"
    if is_binary:
        metrics_name = "Binary Metrics"
        matrix_name = "Binary Confusion Matrix"
    df_final.to_excel(writer, sheet_name=metrics_name, index=False)
    confusion_matrix_df.to_excel(writer, sheet_name=matrix_name, index=False) 
        
        
def save_logits(bag_onehot_labels, bag_logits, class_labels, wsi_names, output_path):
    # 创建 DataFrame 保存数据
    data = {
        "wsi_name": wsi_names,
        "bag_label": bag_onehot_labels,
        "bag_logits": list(bag_logits)  # 将 logits 数组转换为列表
    }
    bag_labels = np.argmax(bag_onehot_labels, axis=1)
    bag_pred = np.argmax(bag_logits, axis=1)
    error_path = os.path.join(os.path.dirname(output_path), "error_log.txt")
    with open(error_path, "w") as f:
        for i in range(len(bag_labels)):
            if bag_labels[i] != bag_pred[i] and bag_labels[i] != 0:
                f.write(f"Error: {wsi_names[i]} label: {class_labels[bag_labels[i]]} \n pred: {bag_logits[i]}\n")
        f.close()
    
    df = pd.DataFrame(data)
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 保存为 CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
        
    
def calc_iou(a, b):

    a=a.type(torch.cuda.DoubleTensor)
    b=b.type(torch.cuda.DoubleTensor)

    area = (b[:, 2] - b[:, 0]+1) * (b[:, 3] - b[:, 1]+1)

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])+1
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])+1

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]+1) * (a[:, 3] - a[:, 1]+1), dim=1) + area - iw * ih

    #ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def collate_fn_wsi(batch):
    """
    参数：
        batch (list): features, label, task_id, file_path, mask [N, M 256]
    返回:
        padded_patches_tensor: [max_N, max_M, C]
        masks_tensor: [max_N, Max_M]
    """
    wsis = [item[0] for item in batch] #each wsi is a list, contain N patches, each patch is a ndarray, with shape [M, 256]
    max_N = min(max([len(wsi) for wsi in wsis]), 1000)
    max_M = max([max([patch.shape[0] for patch in wsi]) for wsi in wsis])

    padded_patches_lists = []
    masks_lists = []
    for wsi in wsis:
        if len(wsi) > max_N:
            wsi = wsi[:max_N] #大于max_N的进行截断
        padded_patches = []
        masks = []
        for patch in wsi:
            # 计算需要 padding 的数量
            padding_size = max_M - patch.shape[0]
            # 对 patch 进行 padding
            
            #padded_patch = F.pad(torch.tensor(patch, dtype=torch.float32), (0, 0, padding_size, 0), mode='constant', value=0)
            #padded_patch = F.pad(patch.clone().detach(), (0, 0, padding_size, 0), mode='constant', value=0)
            padded_patch = F.pad(patch.clone(), (0, 0, padding_size, 0), mode='constant', value=0)
            padded_patches.append(padded_patch)
            # 创建 mask，真实数据部分为 1，padding 部分为 0
            mask = torch.ones(patch.shape[0], dtype=torch.float32)
            if padding_size > 0:
                mask = torch.cat([mask, torch.zeros(padding_size, dtype=torch.float32)], dim=0)
            masks.append(mask)
        # 将 padded patches 和 masks 转换为张量
        padded_patches_lists.append(torch.stack(padded_patches))
        masks_lists.append(torch.stack(masks))
        # padded_patches = [torch.nn.functional.pad(torch.tensor(patch, dtype=torch.float32), (0, 0, 0, max_M - patch.shape[0]), "constant", 0) for patch in wsi]
        # padded_patches_lists.append(padded_patches)

    features = pad_sequence(padded_patches_lists, batch_first=True, padding_value=0) # shape [B, max_N, max_M, 256]
    mask = pad_sequence(masks_lists, batch_first=True, padding_value=0) # shape [B, max_N, max_M]
    
    label = torch.tensor([item[1] for item in batch])
    task_id = torch.tensor([item[2] for item in batch])
    file_path = [item[3] for item in batch]
    
    return features, label, task_id, file_path, mask


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

import logging
import os

_LOGGER_CONFIGURED = False

def print_and_log(message, log_file='output.log', no_print_it=False, level='info'):
    """
    同時打印消息到控制台並記錄到日誌文件，已針對 DDP 環境優化。

    核心改進：
    1.  **避免重複設定**：日誌記錄器 (logger) 只會被設定一次。後續的呼叫會重用已存在的設定，避免了效能問題和潛在衝突。
    2.  **文件路徑鎖定**：日誌檔案的路徑由第一次呼叫時的 `log_file` 參數決定。
    3.  **進程控制**：`print_it` 參數用於控制是否執行打印和記錄。在 DDP 中，你可以在外部邏輯中只對主進程 (rank 0) 將此參數設為 True。
    4.  **輸出統一**：透過設定兩個日誌處理器（一個用於檔案，一個用於控制台），我們可以用一次日誌呼叫完成兩項任務，並避免了重複打印的問題。

    參數:
        message (str): 要輸出的消息內容。
        log_file (str): 日誌文件路徑。只在第一次呼叫時有效，後續呼叫將沿用第一次的設定。
        print_it (bool): 是否執行打印和記錄。設為 False 時，此函數不執行任何操作。
        level (str): 日誌級別 ('debug', 'info', 'warning', 'error', 'critical')。
    """
    global _LOGGER_CONFIGURED

    # 根據外部邏輯 (args.no_log)，如果不需要記錄，則直接返回。
    # 這是 DDP 環境下控制日誌輸出最關鍵的一步。
    if no_print_it:
        return

    # --- 執行一次性的日誌記錄器設定 ---
    # 這段程式碼在每個進程中只會執行一次
    if not _LOGGER_CONFIGURED:
        # 獲取根記錄器
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # 清除可能已存在的 handlers，避免在某些環境(如Jupyter)中重複輸出
        if logger.hasHandlers():
            logger.handlers.clear()

        # 確保日誌目錄存在
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 1. 建立一個 handler 用於寫入日誌檔案
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # 2. 建立一個 handler 用於輸出到控制台 (取代 print())
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        _LOGGER_CONFIGURED = True

    # --- 執行記錄 ---
    # 透過 logging 模組的 level 來記錄，它會自動分發到上面設定好的所有 handlers
    log_levels = {
        'debug': logging.debug,
        'info': logging.info,
        'warning': logging.warning,
        'error': logging.error,
        'critical': logging.critical
    }
    
    log_func = log_levels.get(level.lower(), logging.info)
    log_func(message)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False,save_best_model_stage=0.):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.save_best_model_stage = save_best_model_stage

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        
        score = -val_loss if epoch >= self.save_best_model_stage else 0.

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def state_dict(self):
        return {
            'patience': self.patience,
            'stop_epoch': self.stop_epoch,
            'verbose': self.verbose,
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'val_loss_min': self.val_loss_min
        }
    def load_state_dict(self,dict):
        self.patience = dict['patience']
        self.stop_epoch = dict['stop_epoch']
        self.verbose = dict['verbose']
        self.counter = dict['counter']
        self.best_score = dict['best_score']
        self.early_stop = dict['early_stop']
        self.val_loss_min = dict['val_loss_min']

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

