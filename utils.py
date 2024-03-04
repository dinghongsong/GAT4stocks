import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler



class DailyBatchSampler(Sampler):
    def __init__(self, data):
        self.data = data
        self.daily_count = data.y.groupby("date").size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)  # calculate begin index of each batch
        self.daily_index[0] = 0

    def __iter__(self):
        for idx, count in zip(self.daily_index, self.daily_count):
            yield np.arange(idx, idx + count)

    def __len__(self):
        return len(self.data)


class TabularDataset(Dataset):
    def __init__(self, X: pd.DataFrame, y:pd.Series):
        self.X = X
        self.y = y
        self.X_tensor = torch.tensor(X.values).float()
        self.y_tensor = torch.tensor(y.values).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.X_tensor[item, :], self.y_tensor[item]
    


