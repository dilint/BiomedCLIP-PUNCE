from torchvision.datasets import CIFAR10
import pickle
from PIL import Image
from typing import Any, Callable, Optional, Tuple
import torch
import numpy as np

class CIFAR10Pair(CIFAR10):
    """Generate mini-batche pairs on CIFAR10 training set."""
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair
    
    
class CIFAR10PU(CIFAR10):

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        labeled: int = 1000,
        unlabeled: int = 49000,
    ) -> None:

        super().__init__(root=root, train=train, transform=transform, target_transform=target_transform,
                        download=download )

        self.targets = self._binarize_cifar10_class(self.targets)
        self.labeled = labeled
        self.unlabeled = unlabeled
        
        if self.train :
            self.data, self.targets, self.prior = self._make_pu_label_from_binary_label(self.data, self.targets)

        self._load_meta()

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        img = Image.fromarray(img)  # .convert('RGB')
        imgs = [self.transform(img), self.transform(img)]
        return torch.stack(imgs), target  # stack a positive pair
    
   
    ## add function ##
    def _binarize_cifar10_class(self, y):
        """将类别分为animal和vehicle"""
        # 先转化为numpy
        y = np.array(y)
        y_bin = np.ones(len(y), dtype=int)
        y_bin[(y == 2) | (y == 3) | (y == 4) | (y == 5) | (y == 6) | (y == 7)] = -1
        return y_bin
    
    def _make_pu_label_from_binary_label(self, x, y):
        """挑选出一定的正样本数作为已标注标签"""
        """from https://github.com/kiryor/nnPUlearning"""
        y = np.array(y)
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        labeled, unlabeled = self.labeled, self.unlabeled
        assert(len(x) == len(y))
        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]
        n_p = (y == positive).sum()
        n_lp = labeled
        n_n = (y == negative).sum()
        n_u = unlabeled
        if labeled + unlabeled == len(x):
            n_up = n_p - n_lp
        elif unlabeled == len(x):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        _prior = float(n_up) / float(n_u)
        xlp = x[y == positive][:n_lp]
        xup = np.concatenate((x[y == positive][n_lp:], xlp), axis=0)[:n_up]
        xun = x[y == negative]
        x = np.asarray(np.concatenate((xlp, xup, xun), axis=0))
        print(x.shape)
        y = np.asarray(np.concatenate((np.ones(n_lp), -np.ones(n_u))))
        perm = np.random.permutation(len(y))
        x, y = x[perm], y[perm]
        return x, y, _prior
    
if __name__ == '__main__':
    root_path = '/root/commonfile/hjl/cifar10'
    dataset = CIFAR10PU(root_path, train=True)
    print(len(dataset))
    print(dataset[10])
    print(dataset.prior)