from torch.utils.data import Dataset
import numpy as np
import torch

class TimeDateDataset(Dataset):
    def __init__(self, dates):
        # dates = [date.split(",") for date in dates]
        #
        # convert_int = lambda dt: list(map(int, dt))
        # self.dates = [convert_int(date) for date in dates]
        self.dates = dates

        #print(dates)

    def __len__(self):
        return len(self.dates)-1
    
    def __getitem__(self, idx):
        x = np.array(self.dates[idx]).astype(np.float32)
        convert2year = torch.Tensor(
            [1, 12, 12 * 30, 12 * 30 * 24, 12 * 30 * 24 * 60, 12 * 30 * 24 * 60 * 60]).float()
        # x = x/convert2year
        return x, x/convert2year

if __name__ == "__main__":
    pass