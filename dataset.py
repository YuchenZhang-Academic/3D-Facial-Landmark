from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import util
import torch
import torchvision.transforms as transforms

class FaceDataset(Dataset):
    def __init__(self, dataset_dir, seed=None, mode='train', train_val_ratio=0.9, precision=2) -> None:
        super().__init__()
        if seed is None:
            seed = random.randint(0, 65536)
        random.seed(seed)
        self.precision = precision
        self.dataset_dir = dataset_dir
        self.mode = mode
        if self.mode == 'val':
            mode = 'train'
        dirList = os.listdir(self.dataset_dir)
        self.obj_list = []
        self.mark_list = []
        for i in dirList:
            FilePath = self.dataset_dir + i 
            name = os.listdir(FilePath)
            for j in name:
                if '.obj' in j:
                    self.obj_list.append(FilePath + '/' + j)
                else:
                    self.mark_list.append(FilePath + '/' + j)
        self.data_num = len(self.obj_list)
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
        objData = util.getObj(self.obj_list[id])
        landmarkData = util.getLandmark(self.mark_list[id])
        obj, mark = util.norm(objData, landmarkData)
        np.around(obj, self.precision)
        np.around(mark, self.precision)
        pointsCloud = np.zeros((201, 201, 201))
        marks = []
        funcs = [util.findEyeBox, util.findNoseBox, util.findLipsBox, util.findChinBox, util.findRightFaceBox, util.findLeftFaceBox]
        for i in funcs:
            marks.append(i(mark))
        for i in ((obj + 1) * 100).astype(int).T:
            pointsCloud[i[0], i[1], i[2]] = 1 
        pointsCloud = np.array(pointsCloud)
        marks = np.array(marks)
        return torch.tensor(pointsCloud).reshape(1, 201, 201, 201), torch.tensor(marks)

    def __len__(self):
        return len(self.ids)
    
class MyDataset(Dataset):
    def __init__(self, dataset_dir, seed=None, mode='train', train_val_ratio=0.9, precision=2, organ=0) -> None:
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
    dataset = FaceDataset(dataset_dir)
    dataloader = DataLoader(dataset, 1)
    for i in enumerate(dataloader):
        print(i)
        input("press enter to continue")
    dataset_dir = r"/home/zyc/project/Face/Data/"
    dataset = MyDataset(dataset_dir, organ=0)
    dataloader = DataLoader(dataset)
    for i in enumerate(dataloader):
        print(i)
        input("press enter to continue")