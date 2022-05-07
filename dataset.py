import os, torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class DataGenerator(Dataset):
    def __init__(self, Path, classes):
        self.Path_list = Path
        self.classes = classes
        self.Path_list = np.array(self.Path_list)
        self.indexes = np.arange(len(self.Path_list))
        np.random.shuffle(self.indexes)

    def __len__(self):
        return len(self.Path_list)

    def __getitem__(self, index):
        if index == 0:
            np.random.shuffle(self.indexes)

        indexes = self.indexes[index]
        x, y = np.load(self.Path_list[indexes]), self.classes[indexes]

        return torch.from_numpy(x), y

if __name__ == '__main__':
    data = DataGenerator(os.listdir('I:\\npy_full'))