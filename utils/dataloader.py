import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def simeple_tokenizer(s):
    yield(list(s))


class UniporotBERTDataset(Dataset):
    def __init__(self, root_dir, csv_file, delimiter=',', tokenizer=None):

        self.proteins = pd.read_csv(os.path.join(root_dir, csv_file), delimiter=delimiter)
        self.root_dir = root_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        proteins = self.proteins.iloc[idx, ['name1', 'name2', 'label']]
        encoded = self.tokenizer(sentence_batch, return_tensors='pt', padding=True, truncation=True)
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample