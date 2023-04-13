import sys
from typing import Tuple
from random import randint
# Custom imports
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from util.loader import Data

Sample_t = Tuple[torch.Tensor, torch.Tensor]


class DataSet(TorchDataset):
    """Custom dataset implementation"""

    augment = None

    def sample(self):
        # Randomly return an item as the sample
        idx = randint(0, self.__len__() - 1)
        data, truth, idx = self.__getitem__(idx)
        h, w = data.shape
        data = data.view((-1, h, w))
        w, = truth.shape
        truth = truth.view((-1, w))
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
