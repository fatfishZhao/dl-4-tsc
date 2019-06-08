import numpy as np
import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import logging
import datetime
import cv2

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
def get_train_val_dataset(x_train, x_val, y_train, y_val, transforms = None):

    train_dataset = dataset(x_train,y_train)
    val_dataset = dataset(x_val, y_val)
    return train_dataset, val_dataset


class dataset(data.Dataset):
    def __init__(self, x_array, y_array, transforms=None):
        self.data = x_array
        self.labels = y_array
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        data = self.data[item]
        label = np.argmax(self.labels[item])

        data = data[:, :, np.newaxis]
        data = np.concatenate((data, data, data), axis=2)
        data = data.transpose((2, 0, 1))

        if self.transforms is not None:
            data = self.transforms(data)

        return torch.from_numpy(data).float(), label


def collate_fn(batch):
    datas = []
    label = []

    for sample in batch:
        datas.append(sample[0])
        label.append(sample[1])

    return torch.stack(datas, 0), \
           label