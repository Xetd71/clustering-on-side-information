import torch
from torch.utils.data import Dataset


class SideInformationDataset(Dataset):
    def __init__(self, data, side_information, min_th, max_th):
        self.data = data
        self.side_information = side_information
        self.min_th = min_th
        self.max_th = max_th

    def __len__(self):
        return len(self.inp_data)

    def __getitem__(self, index):
        # target = self.out_data[ind]
        # data_val = self.data[index] [:-1]
        return self.inp_data[index], self.out_data[index]
