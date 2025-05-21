import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score,recall_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from prettytable import PrettyTable
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

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


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


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

def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    try:
        auc_value = roc_auc_score(bag_labels, bag_predictions)
    except Exception as e:
        print(e)
        auc_value = 0
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = accuracy_score(bag_labels, bag_predictions)
    # accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore


def six_scores(bag_labels, bag_predictions, thres):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    if thres != 0:
        threshold_optimal = thres
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = accuracy_score(bag_labels, bag_predictions)
    specificity_metric = BinarySpecificity()
    specificity = specificity_metric(torch.tensor(bag_predictions), torch.tensor(bag_labels))
    # accuracy = 1- np.count_nonzero(np.array(bag_labels).astype(int)- bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, specificity, fscore

def multi_class_scores_mtl(gt_logtis, pred_logits, class_labels, wsi_names, threshold):
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
    
    assert len(class_labels) == 5 or len(class_labels) == 10 or len(class_labels) == 9
    bag_labels = []
    new_pred_logits = []
    
    for i, gt_logit in enumerate(gt_logtis):
        gt_labels = torch.where(gt_logit == 1)[0]
        if len(gt_labels) > 1:
            for gt_label in gt_labels:
                pred_logit = copy.deepcopy(pred_logits[i])
                pred_logit[gt_logit == 1] = 0
                bag_labels.append(gt_label)
                pred_logit[gt_label] = pred_logits[i][gt_label]
                new_pred_logits.append(pred_logit)
        else:
            bag_labels.append(gt_labels[0])
            new_pred_logits.append(pred_logits[i])
            
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(new_pred_logits)
    
    # 对于宫颈癌症风险和微生物感染任务 分开计算指标
    if len(class_labels) in [9, 10]:
        bag_labels_cancer, bag_labels_microbial = bag_labels[bag_labels < 6], bag_labels[bag_labels >= 6]
        bag_logits_cancer, bag_logtis_microbial = bag_logits[bag_labels < 6, :6], bag_logits[bag_labels >= 6, 6:]
        class_labels_cancer, class_labels_microbial = class_labels[:6], class_labels[6:]
    else:
        bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
 
    roc_auc = []
    # 首先评估宫颈癌症风险
    n_cancer_class = len(class_labels_cancer)
    n_cancer_sample = bag_labels_cancer.shape[0]
    # bag_labels_cancer_onehot = np.eye(n_cancer_class)[bag_labels_cancer]
    bag_labels_cancer_onehot = np.array([id2labelcode[l] for l in bag_labels_cancer])
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)
    for i in range(1, n_cancer_class):
        roc_auc.append(roc_auc_score(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i]))
        bag_pred_cancer_onehot[:, i] = bag_logits_cancer[:, i] >= threshold
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
            bag_pred_cancer_onehot[j] = 0
            indices = np.where(bag_logits_cancer[j, 1:] == 1)[0]
            if len(indices) > 0:
                # 根据 rank 值选择优先级最高的索引
                selected_index = max(indices, key=lambda x: rank[x])
                bag_pred_cancer_onehot[j, selected_index+1] = 1
            
            # 如果该类别被判定为NILM的概率过高 也输出错误信息
            if bag_logits_cancer[j, 0] > 0.95:
                print(f'[ERROR] {wsi_names[j]} risk prediction is wrong: {[round(risk, 4) for risk in bag_logits_cancer[j]]}')
    
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracy = accuracy_score(bag_labels_cancer, bag_pred_cancer)
    recalls = recall_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    precisions = precision_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    fscores = f1_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    print('[INFO] confusion matrix for cancer labels:')
    cancer_matrix = confusion_matrix(bag_labels_cancer, bag_pred_cancer, class_labels_cancer)
    print('fscores len' + str(len(fscores)))
    
    # 评估微生物感染
    microbial_matrix = None
    if len(class_labels) in [9, 10]:
        n_microbial_class = len(class_labels_microbial)
        n_microbial_sample = bag_labels_microbial.shape[0]
        bag_labels_microbial = bag_labels_microbial - 6
        bag_labels_microbial_onehot = np.eye(n_microbial_class)[bag_labels_microbial]
        bag_pred_microbial_onehot = np.zeros_like(bag_logtis_microbial)
        for i in range(n_microbial_class):
            roc_auc.append(roc_auc_score(bag_labels_microbial_onehot[:, i], bag_logtis_microbial[:, i]))
            bag_pred_microbial_onehot[:, i] = bag_logtis_microbial[:, i] >= threshold
        for j in range(bag_pred_microbial_onehot.shape[0]):
            if np.sum(bag_pred_microbial_onehot[j]) == 0:
                bag_pred_microbial_onehot[j, 0] = 1
            elif np.sum(bag_pred_microbial_onehot[j]) > 1:
                # 多个类别都大于阈值，则保留得分最高的类别
                bag_pred_microbial_onehot[j] = 0
                bag_pred_microbial_onehot[j, np.argmax(bag_logtis_microbial[j,])] = 1
        bag_pred_microbial = np.argmax(bag_pred_microbial_onehot, axis=-1) # [N,]
        # print(recalls, recalls.shape, type(recalls))
        recalls2 = recall_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        precisions2 = precision_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        fscores2 = f1_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        # print(recalls2, recalls2.shape, type(recalls2))
        recalls, precisions, fscores = np.concatenate((recalls, recalls2)), np.concatenate((precisions, precisions2)), np.concatenate((fscores, fscores2))
        print('[INFO] confusion matrix for microbial labels:')
        microbial_matrix = confusion_matrix(bag_labels_microbial, bag_pred_microbial, class_labels_microbial)
        accuracy_2 = accuracy_score(bag_labels_microbial, bag_pred_microbial)
        accuracy_all = (accuracy * n_cancer_sample + accuracy_2 * n_microbial_sample) / (n_cancer_sample + n_microbial_sample)
        accuracys = [accuracy, accuracy_2, accuracy_all]
    print('Recalls: ' + str(recalls))
    print('roc', 'acc', 'recall', 'prec', 'fs')
    print(roc_auc, accuracys, recalls, precisions, fscores)
    return roc_auc, accuracys, recalls, precisions, fscores, cancer_matrix, microbial_matrix
    # return roc_auc_macro, accuracy, recall, precision, fscore
    
    
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


import pandas as pd
import numpy as np

def save_metrics_to_excel(roc_auc, accuracies, recalls, precisions, fscores, confusion_matrix_cancer_pt, confusion_matrix_microbial_pt, class_labels, output_excel_path):
    """
    将每个类别的AUC、召回率、精确率、F1分数，以及宫颈癌类别和微生物感染类别的平均指标存储到Excel表格中。
    所有指标以百分比形式显示，并保留两位小数。同时保存混淆矩阵（PrettyTable 格式）。
    
    参数：
        roc_auc (list): 每个类别的AUC值。
        accuracies (list): 三个准确率值，分别是宫颈癌类别、微生物感染类别和所有类别的准确率。
        recalls (list): 每个类别的召回率。
        precisions (list): 每个类别的精确率。
        fscores (list): 每个类别的F1分数。
        confusion_matrix_cancer_pt (PrettyTable): 宫颈癌类别的混淆矩阵（PrettyTable 格式）。
        confusion_matrix_microbial_pt (PrettyTable): 微生物感染类别的混淆矩阵（PrettyTable 格式）。
        class_labels (list): 类别标签。
        output_excel_path (str): 输出Excel文件的路径。
    """
    # 将指标转换为百分比形式，并保留两位小数
    roc_auc = [round(auc * 100, 2) for auc in roc_auc]
    recalls = [round(recall * 100, 2) for recall in recalls]
    precisions = [round(precision * 100, 2) for precision in precisions]
    fscores = [round(fscore * 100, 2) for fscore in fscores]
    accuracies = [round(acc * 100, 2) for acc in accuracies]

    # 创建一个DataFrame存储每个类别的指标
    results = {
        "Class": class_labels[1:],
        "AUC (%)": roc_auc,
        "Recall (%)": recalls,
        "Precision (%)": precisions,
        "F1 Score (%)": fscores
    }
    df = pd.DataFrame(results)

    # 计算宫颈癌类别和微生物感染类别的平均指标
    cancer_avg_auc = round(np.mean(roc_auc[:5]), 2)  # 前五个类别为宫颈癌
    cancer_avg_recall = round(np.mean(recalls[:5]), 2)
    cancer_avg_precision = round(np.mean(precisions[:5]), 2)
    cancer_avg_fscore = round(np.mean(fscores[:5]), 2)
    cancer_accuracy = accuracies[0]  # 宫颈癌类别的准确率

    microbial_avg_auc = round(np.mean(roc_auc[5:]), 2)  # 后面几个类别为微生物感染
    microbial_avg_recall = round(np.mean(recalls[5:]), 2)
    microbial_avg_precision = round(np.mean(precisions[5:]), 2)
    microbial_avg_fscore = round(np.mean(fscores[5:]), 2)
    microbial_accuracy = accuracies[1]  # 微生物感染类别的准确率

    # 计算所有类别的平均指标
    all_avg_auc = round(np.mean(roc_auc), 2)
    all_avg_recall = round(np.mean(recalls), 2)
    all_avg_precision = round(np.mean(precisions), 2)
    all_avg_fscore = round(np.mean(fscores), 2)
    all_accuracy = accuracies[2]  # 所有类别的准确率

    # 将平均指标添加到DataFrame中
    df_avg = pd.DataFrame({
        "Class": ["Cervical Cancer Average", "Microbial Infection Average", "All Classes Average"],
        "AUC (%)": [cancer_avg_auc, microbial_avg_auc, all_avg_auc],
        "Recall (%)": [cancer_avg_recall, microbial_avg_recall, all_avg_recall],
        "Precision (%)": [cancer_avg_precision, microbial_avg_precision, all_avg_precision],
        "F1 Score (%)": [cancer_avg_fscore, microbial_avg_fscore, all_avg_fscore],
        "Accuracy (%)": [cancer_accuracy, microbial_accuracy, all_accuracy]
    })

    # 合并结果
    df_final = pd.concat([df, df_avg], ignore_index=True)

    # 将 PrettyTable 转换为 DataFrame
    confusion_matrix_cancer_df = prettytable_to_dataframe(confusion_matrix_cancer_pt)
    confusion_matrix_microbial_df = prettytable_to_dataframe(confusion_matrix_microbial_pt)

    # 将混淆矩阵保存到Excel的不同Sheet中
    with pd.ExcelWriter(output_excel_path) as writer:
        df_final.to_excel(writer, sheet_name="Metrics", index=False)
        confusion_matrix_cancer_df.to_excel(writer, sheet_name="Confusion Matrix (Cancer)", index=False)
        confusion_matrix_microbial_df.to_excel(writer, sheet_name="Confusion Matrix (Microbial)", index=False)

    print(f"Metrics and confusion matrices saved to {output_excel_path}")
        
def save_logits(bag_onehot_labels, bag_logits, class_labels, wsi_names, output_path):
    # 创建 DataFrame 保存数据
    data = {
        "wsi_name": wsi_names,
        "bag_label": bag_onehot_labels,
        "bag_logits": list(bag_logits)  # 将 logits 数组转换为列表
    }
    
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
from datetime import datetime

def print_and_log(message, log_file='output.log', print_it=True, level='info'):
    """
    同时打印消息和记录到日志文件
    
    参数:
        message: 要输出的消息内容
        log_file: 日志文件路径(默认'output.log')
        print_it: 是否打印到控制台(默认True)
        level: 日志级别('debug', 'info', 'warning', 'error', 'critical'，默认'info')
    """
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'  # 追加模式
    )
    
    # 打印到控制台
    if print_it:
        print(message)
    
    # 记录到日志文件
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

