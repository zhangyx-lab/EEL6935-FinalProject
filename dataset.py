# ---------------------------------------------------------
# Yuxuan Zhang
# Dept. of Electrical and Computer Engineering
# University of Florida
# ---------------------------------------------------------
# Dataset wrapper for Ken Natural Images
# ---------------------------------------------------------
import sys
from typing import Tuple
from random import randint, shuffle
# Custom imports
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from util.loader import Data

Sample_t = Tuple[torch.Tensor, torch.Tensor]


class DataSet(TorchDataset):
    """Custom dataset implementation"""

    augment = None

    def sample(self, batchSize=1):
        # Randomly return an item as the sample
        idx = list(range(self.__len__()))
        shuffle(idx)
        idx = idx[:batchSize]
        result = [self.__getitem__(i) for i in idx]
        data, truth, idx = zip(*result)
        data = torch.stack(data, dim=0)
        truth = torch.stack(truth, dim=0)
        return data, truth


    def __init__(self, data: Data):
        """
        load dataset from a given data set collection
        """
        self.data = data


    def __len__(self):
        return len(self.data.labels)


    def __getitem__(self, idx):
        data = self.data.stimuli[idx]
        truth = self.data.responses[idx]
        labels = self.data.labels[idx]
        data = torch.from_numpy(data.astype(np.float32))
        truth = torch.from_numpy(truth.astype(np.float32))
        return data, truth, idx
