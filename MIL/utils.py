import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score,recall_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from torchmetrics.classification import BinarySpecificity, BinaryRecall
from prettytable import PrettyTable

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

def multi_class_scores_nonilm(bag_labels, bag_logits, class_labels):
    # 去掉NILM类别 ，通过每个类别分别计算tpr和fpr，获得optimal_threshold，如果每个类别都为0，则被分为NILM
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(bag_logits)
    n_classes = max(bag_labels) + 1
    bag_labels_one_hot = np.eye(n_classes)[bag_labels]
    
    roc_auc = dict()
    bag_pred_onehot = np.zeros_like(bag_logits)
    thresholds = []
    for i in range(1, n_classes):
        roc_auc[i] = roc_auc_score(bag_labels_one_hot[:, i], bag_logits[:, i])
        fpr, tpr, threshold = roc_curve(bag_labels_one_hot[:, i], bag_logits[:, i], pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        thresholds.append(threshold_optimal)
        bag_pred_onehot[:, i] = bag_logits[:, i] >= threshold_optimal
    print("Info: thresholds for 4 positive classes:")
    print(thresholds)
    for j in range(bag_pred_onehot.shape[0]):
        if np.sum(bag_pred_onehot[j]) == 0:
            bag_pred_onehot[j, 0] = 1
        elif np.sum(bag_pred_onehot[j]) > 1:
            # 多个类别都大于阈值，则保留最大概率的类别
            bag_pred_onehot[j] = 0
            bag_pred_onehot[j, np.argmax(bag_logits[j, 1:])+1] = 1
    bag_pred = np.argmax(bag_pred_onehot, axis=-1)
    
    roc_auc = list(roc_auc.values())
    roc_auc_macro = np.mean(roc_auc)
    
    accuracy = accuracy_score(bag_labels, bag_pred)
    recall = recall_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    precision = precision_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    fscore = f1_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    two_class_scores(bag_labels, bag_pred)
    confusion_matrix(bag_labels, bag_pred_onehot, class_labels)
    return roc_auc_macro, accuracy, recall, precision, fscore

def multi_class_scores_nonilmv2(bag_labels, bag_logits, class_labels, wsi_names, eval_only):
    # 去掉NILM类别 ，通过每个类别分别计算tpr和fpr，获得optimal_threshold，如果每个类别都为0，则被分为NILM
    eval_method=1
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(bag_logits)
    n_classes = max(bag_labels) + 1
    bag_labels_one_hot = np.eye(n_classes)[bag_labels]
    
    roc_auc = dict()
    bag_pred_onehot = np.zeros_like(bag_logits)
    thresholds, num_pos = [], []
    threshold_set = [0, 0.3, 0.3, 0.3, 0.3]
    for i in range(1, n_classes):
        roc_auc[i] = roc_auc_score(bag_labels_one_hot[:, i], bag_logits[:, i])
        fpr, tpr, threshold = roc_curve(bag_labels_one_hot[:, i], bag_logits[:, i], pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        threshold_optimal = threshold_set[i]
        bag_pred_onehot[:, i] = bag_logits[:, i] >= threshold_optimal
        thresholds.append(threshold_optimal)
        num_pos.append(np.sum(bag_pred_onehot[:, i]))
    print("Info: thresholds for 4 positive classes:")
    print(thresholds)
    print("Info: positive num for 4 positive classes:")
    print(num_pos)
    for j in range(bag_pred_onehot.shape[0]):
        if np.sum(bag_pred_onehot[j]) == 0:
            bag_pred_onehot[j, 0] = 1
        elif np.sum(bag_pred_onehot[j]) > 1:
            # 多个类别都大于阈值，则保留最大风险的类别
            if eval_method == 1:
                for i, score in enumerate(bag_pred_onehot[j]):
                    if np.sum(bag_pred_onehot[j]) == 0:
                        bag_pred_onehot[j, 0] = 1
                    elif np.sum(bag_pred_onehot[j]) > 1:
                        # 多个类别都大于阈值，则保留最大概率的类别
                        bag_pred_onehot[j] = 0
                        bag_pred_onehot[j, np.argmax(bag_logits[j, 1:])+1] = 1
            # 多个类别都大于阈值，则保留最大概率的类别
            elif eval_method == 2:
                for i, score in enumerate(bag_pred_onehot[j]):
                    if score >= 1 and i > 0:
                        class_index = i 
                bag_pred_onehot[j] = 0
                bag_pred_onehot[j, class_index] = 1
    bag_pred = np.argmax(bag_pred_onehot, axis=-1)
    
    roc_auc = list(roc_auc.values())
    roc_auc_macro = np.mean(roc_auc)
    
    accuracy = accuracy_score(bag_labels, bag_pred)
    recall = recall_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    precision = precision_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    fscore = f1_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    # 方便查看错误样本
    if eval_only:
        err_pos_count=0
        for i, wsi_name in enumerate(wsi_names):
            # if bag_pred[i] == bag_labels[i]:
            #     print(f"[Info]: correct class, wsi_name: {wsi_name}, labe;: {bag_labels[i]}")
            #     continue
            # elif bag_labels[i] == 0:
            #     print(f"[Info]: error class for negative sample, wsi_name: {wsi_name}, label: {bag_labels[i]}, perd: {bag_pred[i]}")
            # else:
            #     err_pos_count += 1
            #     print(f"[Warning]{err_pos_count}: wsi_name: {wsi_name}, label: {bag_labels[i]}, perd: {bag_pred[i]}")
            if bag_labels[i] == 4:
                print(f"[Info] Logits for {wsi_name}: {[round(bag_logit, 4) for bag_logit in bag_logits[i]]}")
        print(f'Accuracy: {round(accuracy,4)}, Recall: {round(recall,4)}, Precision: {round(precision,4)}, Fscore: {round(fscore,4)}, ROC_AUC: {round(roc_auc_macro,4)}')
    
    # 打印混淆矩阵
    two_class_scores(bag_labels, bag_pred)
    confusion_matrix(bag_labels, bag_pred_onehot, class_labels)
    
    return roc_auc_macro, accuracy, recall, precision, fscore

def multi_class_scores_mtl(bag_labels, bag_logits, class_labels, wsi_names, threshold, eval_only):
    """
    参数：
        bag_labels (list): N, 真实标签
        bag_logtis (tensor): [N, num_class], 每个样本的类别概率
        class_labels (list): [['NILM'], ['ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC'], ['BV', 'M', 'T']],  类别标签, 
        wsi_names (list): N, WSI名称,方便打印错误信息
        eval_only (bool): 是否只进行评估
    返回：
        roc_auc_macro (ndarray): 多类别ROC_AUC
        accuracy (float): Micro 准确率
        recall (ndarray): Macro 阳性召回率 len = 8 or 4 
        precision (ndarray): Macro 阳性精确率
        fscore (ndarray): Macro F1分数
    TODO 目前对于多类别任务，只考虑了1,5,3的多分类划分方式以及5分类的单任务模式
    """
    # 将所有类别合并处理
    assert len(class_labels) == 5 or len(class_labels) == 9 
    bag_labels = np.array(bag_labels)
    bag_logits = np.array(bag_logits)
 
    # 对于宫颈癌症风险和微生物感染任务 分开计算指标
    if len(class_labels) == 9:
        bag_labels_cancer, bag_labels_microbial = bag_labels[bag_labels < 6], bag_labels[bag_labels >= 6]
        bag_logits_cancer, bag_logtis_microbial = bag_logits[bag_labels < 6, :], bag_logits[bag_labels >= 6, :]
        class_labels_cancer, class_labels_microbial = class_labels[:6], class_labels[6:]
    else:
        bag_labels_cancer, bag_logits_cancer, class_labels_cancer = bag_labels, bag_logits, class_labels
 
    roc_auc = []
    # 首先评估宫颈癌症风险
    n_cancer_class = len(class_labels_cancer)
    n_cancer_sample = bag_labels_cancer.shape[0]
    bag_labels_cancer_onehot = np.eye(n_cancer_class)[bag_labels_cancer]
    bag_pred_cancer_onehot = np.zeros_like(bag_logits_cancer)
    for i in range(1, n_cancer_class):
        roc_auc.append(roc_auc_score(bag_labels_cancer_onehot[:, i], bag_logits_cancer[:, i]))
        bag_pred_cancer_onehot[:, i] = bag_logits_cancer[:, i] >= threshold
    # print(bag_pred_cancer_onehot.shape)
    for j in range(bag_pred_cancer_onehot.shape[0]):
        if np.sum(bag_pred_cancer_onehot[j]) == 0:
            bag_pred_cancer_onehot[j, 0] = 1
        elif np.sum(bag_pred_cancer_onehot[j]) > 1:
            # 多个类别都大于阈值，则保留最大风险的类别
            bag_pred_cancer_onehot[j] = 0
            bag_pred_cancer_onehot[j, np.argmax(bag_logits_cancer[j, 1:])+1] = 1
            # 如果该类别被判定为NILM的概率过高 也输出错误信息
            if bag_logits_cancer[j, 0] > 0.95:
                print(f'[ERROR] {wsi_names[j]} risk prediction is wrong: {[round(risk, 4) for risk in bag_logits_cancer[j]]}')
    
    bag_pred_cancer = np.argmax(bag_pred_cancer_onehot, axis=-1) # [N_cancer,]
    accuracy = accuracy_score(bag_labels_cancer, bag_pred_cancer)
    recalls = recall_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    precisions = precision_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    fscores = f1_score(bag_labels_cancer, bag_pred_cancer, average=None, labels=list(range(1,n_cancer_class)))
    print('[INFO] confusion matrix for cancer labels:')
    confusion_matrix(bag_labels_cancer, bag_pred_cancer, class_labels_cancer)
    print('fscores len' + str(len(fscores)))
    
    # 评估微生物感染
    if len(class_labels) == 9:
        n_microbial_class = len(class_labels_microbial)
        n_microbial_sample = bag_labels_microbial.shape[0]
        bag_labels_microbial_onehot = np.eye(n_microbial_class)[bag_labels_microbial-6]
        bag_pred_microbial_onehot = np.zeros_like(bag_logtis_microbial)
        for i in range(n_microbial_class):
            roc_auc.append(roc_auc_score(bag_labels_microbial_onehot[:, i], bag_logtis_microbial[:, i]))
            bag_pred_microbial_onehot[:, i] = bag_logtis_microbial[:, i] >= threshold
        for j in range(bag_pred_microbial_onehot.shape[0]):
            if np.sum(bag_pred_microbial_onehot[j]) == 0:
                bag_pred_microbial_onehot[j, 0] = 1
            elif np.sum(bag_pred_microbial_onehot[j]) > 1:
                # 多个类别都大于阈值，则保留最大风险的类别
                bag_pred_microbial_onehot[j] = 0
                bag_pred_microbial_onehot[j, np.argmax(bag_logtis_microbial[j, 1:])+1] = 1
        bag_pred_microbial = np.argmax(bag_pred_microbial_onehot, axis=-1) # [N,]
        # print(recalls, recalls.shape, type(recalls))
        recalls2 = recall_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        precisions2 = precision_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        fscores2 = f1_score(bag_labels_microbial, bag_pred_microbial, average=None, labels=list(range(n_microbial_class)))
        # print(recalls2, recalls2.shape, type(recalls2))
        recalls, precisions, fscores = np.concatenate((recalls, recalls2)), np.concatenate((precisions, precisions2)), np.concatenate((fscores, fscores2))
        print('[INFO] confusion matrix for microbial labels:')
        confusion_matrix(bag_labels_microbial-6, bag_pred_microbial, class_labels_microbial)
        accuracy_2 = accuracy_score(bag_labels_microbial, bag_pred_microbial)
        accuracy = (accuracy * n_cancer_sample + accuracy_2 * n_microbial_sample) / (n_cancer_sample + n_microbial_sample)
    print('Recalls: ' + str(recalls))
    return roc_auc, accuracy, recalls, precisions, fscores
    # return roc_auc_macro, accuracy, recall, precision, fscore


def multi_class_scores(bag_labels, bag_logits, class_labels):
    bag_labels = np.array(bag_labels)
    n_classes = max(bag_labels) + 1
    bag_labels_one_hot = np.eye(n_classes)[bag_labels]

    bag_logits = np.array(bag_logits)
    bag_pred = np.argmax(bag_logits, axis=-1)
    accuracy = accuracy_score(bag_labels, bag_pred)
    recall = recall_score(bag_labels, bag_pred, average=None)
    print(recall)
    recall = recall_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    precision = precision_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    fscore = f1_score(bag_labels, bag_pred, average='macro', labels=list(range(1,n_classes)))
    roc_auc = dict()
    for i in range(1, n_classes):
        roc_auc[i] = roc_auc_score(bag_labels_one_hot[:, i], bag_logits[:, i])
    roc_auc = list(roc_auc.values())
    roc_auc_macro = np.mean(roc_auc)
    two_class_scores(bag_labels, bag_pred)
    confusion_matrix(bag_labels, np.eye(n_classes)[bag_pred], class_labels)
    return roc_auc_macro, accuracy, recall, precision, fscore

def two_class_scores(bag_labels, bag_pred):
    bag_labels = [1 if i != 0 else 0 for i in bag_labels]
    bag_pred = [1 if i != 0 else 0 for i in bag_pred]
    bag_labels, bag_pred = np.array(bag_labels), np.array(bag_pred)

    accuracy = accuracy_score(bag_labels, bag_pred)
    print(f"Two class Acc:{accuracy}")
    confusion_matrix(bag_labels, bag_pred, ['NILM', 'POS'])
    
        
def confusion_matrix(bag_labels, bag_logits, class_labels):
    if isinstance(bag_logits[0], np.ndarray):
        y_true, y_pred = bag_labels, np.argmax(np.array(bag_logits), axis=-1)
    else:
        y_true, y_pred = bag_labels, np.array([1 if x > 0.5 else 0 for x in bag_logits])
    num_classes = len(class_labels)

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

