from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import util
import torch
import torchvision.transforms as transforms

class MyDataset(Dataset):
    def __init__(self, dataset_dir, seed=None, mode='train', train_val_ratio=0.9, precision=2, organ=0) -> None:

        '''
        dataset_dir: 数据所处的文件夹
        seed: 随机种子
        mode: 三种模式：train, val, test
        train_val_ratio: 训练时训练集与验证集的比例
        trans: 数据预处理函数

        TODO
        '''
        super().__init__()
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
        self.organ = organ
        self.precision = precision
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.objData, self.landmarkData = util.getOrganPoints(self.dataset_dir, self.organ)
        if self.mode == 'val':
            mode = 'train'
        self.data_num = len(self.objData)
        ids = list(range(self.data_num))
        num_train = int(train_val_ratio * self.data_num)
        if self.mode == 'train':
            self.ids = ids[:num_train]
        elif self.mode == 'val':
            self.ids = ids[num_train:]
        else:
            self.ids = ids
            
        
    def __getitem__(self, item):        
        id = self.ids[item]
        objData = self.objData[id]
        landmarkData = self.landmarkData[id]
        return torch.tensor(objData).reshape(1, 201, 201, 201), torch.tensor(landmarkData)

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    dataset_dir = r"/home/zyc/project/Face/Data/"
    dataset = MyDataset(dataset_dir, organ=0)
    dataloader = DataLoader(dataset)
    for i in enumerate(dataloader):
        print(i)
        input("press enter to continue")