import os
import csv
import torch
import random
import numpy as np
from collections import Counter, defaultdict
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
import math

def readCSV(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines

def get_patient_label(csv_file):
    patients_list=[]
    labels_list=[]
    label_file = readCSV(csv_file)
    for i in range(0, len(label_file)):
        patients_list.append(label_file[i][0])
        labels_list.append(label_file[i][1])
    a=Counter(labels_list)
    print("patient_len:{} label_len:{}".format(len(patients_list), len(labels_list)))
    print("all_counter:{}".format(dict(a)))
    return np.array(patients_list,dtype=object), np.array(labels_list,dtype=object)

def data_split(full_list, ratio, shuffle=True,label=None,label_balance_val=True):
    """
    dataset split: split the full_list randomly into two sublist (val-set and train-set) based on the ratio
    :param full_list: 
    :param ratio:     
    :param shuffle:  
    """
    # select the val-set based on the label ratio
    if label_balance_val and label is not None:
        _label = label[full_list]
        _label_uni = np.unique(_label)
        sublist_1 = []
        sublist_2 = []

        for _l in _label_uni:
            _list = full_list[_label == _l]
            n_total = len(_list)
            offset = int(n_total * ratio)
            if shuffle:
                random.shuffle(_list)
            sublist_1.extend(_list[:offset])
            sublist_2.extend(_list[offset:])
    else:
        n_total = len(full_list)
        offset = int(n_total * ratio)
        if n_total == 0 or offset < 1:
            return [], full_list
        if shuffle:
            random.shuffle(full_list)
        val_set = full_list[:offset]
        train_set = full_list[offset:]

    return val_set, train_set


def get_kflod(k, patients_array, labels_array,val_ratio=False,label_balance_val=True):
    if k > 1:
        skf = StratifiedKFold(n_splits=k)
    else:
        raise NotImplementedError
    train_patients_list = []
    train_labels_list = []
    test_patients_list = []
    test_labels_list = []
    val_patients_list = []
    val_labels_list = []
    for train_index, test_index in skf.split(patients_array, labels_array):
        if val_ratio != 0.:
            val_index,train_index = data_split(train_index,val_ratio,True,labels_array,label_balance_val)
            x_val, y_val = patients_array[val_index], labels_array[val_index]
        else:
            x_val, y_val = [],[]
        x_train, x_test = patients_array[train_index], patients_array[test_index]
        y_train, y_test = labels_array[train_index], labels_array[test_index]

        train_patients_list.append(x_train)
        train_labels_list.append(y_train)
        test_patients_list.append(x_test)
        test_labels_list.append(y_test)
        val_patients_list.append(x_val)
        val_labels_list.append(y_val)
        
    # print("get_kflod.type:{}".format(type(np.array(train_patients_list))))
    return np.array(train_patients_list,dtype=object), np.array(train_labels_list,dtype=object), np.array(test_patients_list,dtype=object), np.array(test_labels_list,dtype=object),np.array(val_patients_list,dtype=object), np.array(val_labels_list,dtype=object)

def get_tcga_parser(root,cls_name,mini=False):
        x = []
        y = []

        for idx,_cls in enumerate(cls_name):
            _dir = 'mini_pt' if mini else 'pt_files'
            _files = os.listdir(os.path.join(root,_cls,'features',_dir))
            _files = [os.path.join(os.path.join(root,_cls,'features',_dir,_files[i])) for i in range(len(_files))]
            x.extend(_files)
            y.extend([idx for i in range(len(_files))])
            
        return np.array(x).flatten(),np.array(y).flatten()

class TCGADataset(Dataset):
    
    def __init__(self, file_name=None, file_label=None,max_patch=-1,root=None,persistence=True,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(TCGADataset, self).__init__()

        self.patient_name = file_name
        self.patient_label = file_label
        self.max_patch = max_patch
        self.root = root
        self.all_pts = os.listdir(os.path.join(self.root,'h5_files')) if keep_same_psize else os.listdir(os.path.join(self.root,'pt_files'))
        self.slide_name = []
        self.slide_label = []
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        for i,_patient_name in enumerate(self.patient_name):
            _sides = np.array([ _slide if _patient_name in _slide else '0' for _slide in self.all_pts])
            _ids = np.where(_sides != '0')[0]
            for _idx in _ids:
                if persistence:
                    self.slide_name.append(torch.load(os.path.join(self.root,'pt_files',_sides[_idx])))
                else:
                    self.slide_name.append(_sides[_idx])
                self.slide_label.append(self.patient_label[i])
        self.slide_label = [ 0 if _l == 'LUAD' else 1 for _l in self.slide_label]

    def __len__(self):
        return len(self.slide_name)

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        file_path = self.slide_name[idx]
        label = self.slide_label[idx]

        if self.persistence:
            features = file_path
        else:
            features = torch.load(os.path.join(self.root,'pt_files',file_path))
        return features , int(label)

class C16Dataset(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(C16Dataset, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            if "pt" in os.listdir(self.root):
                dir_path = os.path.join(self.root,"pt")
            else:
                dir_path = self.root
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path, map_location='cpu')

        label = int(self.slide_label[idx])
        return features, label, file_path
    
class GcMTLDataset(C16Dataset):
    def __init__(self, file_name, file_label,root,persistence,keep_same_psize,num_classes,num_task,is_train=False):
        super(GcMTLDataset, self).__init__(file_name, file_label,root,persistence,keep_same_psize,is_train)
        self.num_classes = num_classes
        self.num_task = num_task
        
    def __getitem__(self, idx):
        features, label, file_path = super().__getitem__(idx)
        tensor_num_classes = torch.tensor(self.num_classes)
        tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0)
        # 第一个位置大于label的位置即为task_id
        task_id = next((i for i, x in enumerate(tensor_num_classes_cumsum) if x > label), -1)

        if task_id > 0:
            label -= tensor_num_classes_cumsum[task_id-1]
        return features, label, task_id, file_path
    
class GcDataset(C16Dataset):
    def __init__(self, file_name, file_label,root,persistence,keep_same_psize,high_weight,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(GcDataset, self).__init__(file_name, file_label,root,persistence,keep_same_psize,is_train)
        self.high_labels = ['ASC-H', 'HSIL']
        self.high_weight = high_weight
        self.is_train = is_train
        
    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = self.root
            if "pt" in os.listdir(self.root):
                dir_path = os.path.join(self.root,"pt")
            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)
        label = int(self.slide_label[idx])
        target = F.one_hot(torch.tensor(label), num_classes=2).type(torch.float32)
        # adapt one_hot to calculate CEloss and if the wsi label is a high risk label, increase the loss weight for this sample  
        for high_label in self.high_labels:
            if high_label in self.file_name[idx]:
                target[1] = self.high_weight
                break
        if self.is_train:
            return features, target, file_path
        else:
            return features, label, file_path

class NGCDatasetInfer(Dataset):

    def __init__(self, file_name, file_label,root,persistence=False,keep_same_psize=0,is_train=False):
        """
        Args
        :param images: 
        :param transform: optional transform to be applied on a sample
        """
        super(NGCDatasetInfer, self).__init__()
        self.file_name = file_name
        self.slide_label = file_label
        self.slide_label = [int(_l) for _l in self.slide_label]
        self.size = len(self.file_name)
        self.root = root
        self.persistence = persistence
        self.keep_same_psize = keep_same_psize
        self.is_train = is_train

        if persistence:
            self.feats = [ torch.load(os.path.join(root,'pt', _f+'.pt')) for _f in file_name ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Args
        :param idx: the index of item
        :return: image and its label
        """
        if self.persistence:
            features = self.feats[idx]
        else:
            dir_path = os.path.join(self.root,"pt")

            file_path = os.path.join(dir_path, self.file_name[idx]+'.pt')
            features = torch.load(file_path)

        label = int(self.slide_label[idx])
        wsi_name = self.file_name[idx]
        
        return features, label, wsi_name
    

# Copyright (c) OpenMMLab. All rights reserved.
# 修改自mmengine的ClassBalancedDataset类
class ClassBalancedDataset(Dataset):
    """A wrapper of class balanced dataset.

    Suitable for training on class imbalanced datasets like LVIS. Following
    the sampling strategy in the `paper <https://arxiv.org/abs/1908.03195>`_,
    in each epoch, an image may appear multiple times based on its
    "repeat factor".
    The repeat factor for an image is a function of the frequency the rarest
    category labeled in that image. The "frequency of category c" in [0, 1]
    is defined by the fraction of images in the training set (without repeats)
    in which category c appears.
    The dataset needs to instantiate :meth:`get_cat_ids` to support
    ClassBalancedDataset.

    The repeat factor is computed as followed.

    1. For each category c, compute the fraction # of images
       that contain it: :math:`f(c)`
    2. For each category c, compute the category-level repeat factor:
       :math:`r(c) = max(1, sqrt(t/f(c)))`
    3. For each image I, compute the image-level repeat factor:
       :math:`r(I) = max_{c in I} r(c)`

    Note:
        ``ClassBalancedDataset`` should not inherit from ``BaseDataset``
        since ``get_subset`` and ``get_subset_`` could  produce ambiguous
        meaning sub-dataset which conflicts with original dataset. If you
        want to use a sub-dataset of ``ClassBalancedDataset``, you should set
        ``indices`` arguments for wrapped dataset which inherit from
        ``BaseDataset``.

    Args:
        dataset (BaseDataset or dict): The dataset to be repeated.
        oversample_thr (float): frequency threshold below which data is
            repeated. For categories with ``f_c >= oversample_thr``, there is
            no oversampling. For categories with ``f_c < oversample_thr``, the
            degree of oversampling following the square-root inverse frequency
            heuristic above.
        lazy_init (bool, optional): whether to load annotation during
            instantiation. Defaults to False
    """

    def __init__(self,
                 dataset: Dataset,
                 oversample_thr: float,
                 lazy_init: bool = False):
        self.dataset = dataset
        self.oversample_thr = oversample_thr
        self._fully_initialized = False
        if not lazy_init:
            self.full_init()

    
    def full_init(self):
        """Loop to ``full_init`` each dataset."""
        if self._fully_initialized:
            return

        # Get repeat factors for each image.
        repeat_factors = self._get_repeat_factors(self.dataset,
                                                  self.oversample_thr)
        # Repeat dataset's indices according to repeat_factors. For example,
        # if `repeat_factors = [1, 2, 3]`, and the `len(dataset) == 3`,
        # the repeated indices will be [1, 2, 2, 3, 3, 3].
        repeat_indices = []
        for dataset_index, repeat_factor in enumerate(repeat_factors):
            repeat_indices.extend([dataset_index] * math.ceil(repeat_factor))
        self.repeat_indices = repeat_indices

        self._fully_initialized = True

    def _get_repeat_factors(self, dataset: Dataset,
                            repeat_thr: float):
        """Get repeat factor for each images in the dataset.

        Args:
            dataset (BaseDataset): The dataset.
            repeat_thr (float): The threshold of frequency. If an image
                contains the categories whose frequency below the threshold,
                it would be repeated.

        Returns:
            List[float]: The repeat factors for each images in the dataset.
        """
        # 1. For each category c, compute the fraction # of images
        #   that contain it: f(c)
        category_freq: defaultdict = defaultdict(float)
        num_images = len(dataset)
        for idx in range(num_images):
            category_freq[self.dataset.slide_label[idx]] += 1
        for k, v in category_freq.items():
            assert v > 0, f'caterogy {k} does not contain any images'
            category_freq[k] = v / num_images

        # 2. For each category c, compute the category-level repeat factor:
        #    r(c) = max(1, sqrt(t/f(c)))
        category_repeat = {
            cat_id: max(1.0, math.sqrt(repeat_thr / cat_freq))
            for cat_id, cat_freq in category_freq.items()
        }

        # 3. For each image I and its labels L(I), compute the image-level
        # repeat factor:
        #    r(I) = max_{c in L(I)} r(c)
        repeat_factors = []
        for idx in range(num_images):
            # the length of `repeat_factors` need equal to the length of
            # dataset. Hence, if the `cat_ids` is empty,
            # the repeat_factor should be 1.
            repeat_factor = category_repeat[self.dataset.slide_label[idx]]
            repeat_factors.append(repeat_factor)

        print(category_repeat)
        return repeat_factors

    def _get_ori_dataset_idx(self, idx: int) -> int:
        """Convert global index to local index.

        Args:
            idx (int): Global index of ``RepeatDataset``.

        Returns:
            int: Local index of data.
        """
        return self.repeat_indices[idx]

    def get_cat_ids(self, idx: int) :
        """Get category ids of class balanced dataset by index.

        Args:
            idx (int): Index of data.

        Returns:
            List[int]: All categories in the image of specified index.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_cat_ids(sample_idx)

    def get_data_info(self, idx: int) -> dict:
        """Get annotation by index.

        Args:
            idx (int): Global index of ``ConcatDataset``.

        Returns:
            dict: The idx-th annotation of the dataset.
        """
        sample_idx = self._get_ori_dataset_idx(idx)
        return self.dataset.get_data_info(sample_idx)

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        ori_index = self._get_ori_dataset_idx(idx)
        return self.dataset[ori_index]

    def __len__(self):
        return len(self.repeat_indices)

