import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch
import os

class LLMDataset(Dataset):
    def __init__(self, data_path, seq_len):
        super().__init__()

        self.seq_len = seq_len
        self.data = np.memmap(data_path, dtype = np.uint16, mode = 'r') #veriyi ram'e alma, diskten anlık oku.

        def __len__(self):
            return len(self.data) - self.seq_len-1
        
        def __getitem__(self, idx):
            chunk = self.data[idx:idx + self.seq_len+1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:], dtype=torch.long)

            return x,y