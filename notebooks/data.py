import os
from pathlib import Path
from typing import List,Optional,Dict,Tuple,Union,Any

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader,Dataset
from torchvision.transforms import Compose,PILToTensor,Normalize

from tqdm import tqdm,trange

import matplotlib.pyplot as plt
import pickle

class Cifar10(Dataset):

    def __init__(self,data_dir:str,train:bool=True,transform=None,label_transform=None):

        self.data_dir = data_dir

        self.Xs = np.array([])
        self.ys = np.array([])
        self.names = {}

        self.transform = transform
        self.label_transform = label_transform

        self.label_names = {
            0: "airplane",
            1: "automobile",
            2: "bird",
            3: "cat",
            4: "deer",
            5: "dog",
            6: "frog",
            7: "horse",
            8: "ship" ,
            9: "truck",
        }

        def filename_to_name_idx(name):
            name = str(name).split('_')
            return name[-1],name[0]

        if train:
            files = filter(lambda file: 'data' in file, os.listdir(self.data_dir))
        else:
            files = filter(lambda file: 'test' in file, os.listdir(self.data_dir))

        count = 0
        for file in files:
           data,labels,names = self.unpickle(os.path.join(self.data_dir,file))
            if count == 0:
                self.Xs = np.array(data)
                self.ys = np.array(labels)
                count += 1
            else:
                self.Xs = np.vstack((self.Xs,data))
                self.ys = np.vstack((self.ys,labels))

            names = {idx:name for idx,name in map(filename_to_name_idx,names)}

            self.names.update(names)

        self.ys = self.ys.reshape(-1)
        assert self.Xs.shape[0]==self.ys.shape[0],f"Data and labels are not in same shape {self.Xs.shape,self.ys.shape}"

    def __len__(self) -> int:
        return self.Xs.shape[0]

    def __getitem__(self,idx) -> Tuple[np.ndarray,Union[int,np.ndarray]]:
        image = self.Xs[idx]
        label = self.ys[idx]
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image,label

    def show_example(self,idx:int) -> None:
        img,label = self.__getitem__(idx)
        plt.imshow(img.transpose(1,2,0))
        plt.title(self.label_names[label])
        plt.axis('off')
        plt.show()

    def show_random_example(self) -> None:
        idx = np.random.randint(0,self.__len__())
        self.show_example(idx)

    def get_random_grid(self,grid_size:int = 5,viz:bool=False) -> np.ndarray:

        idx = lambda : np.random.randint(0,len(self))
        img = np.concatenate([ np.concatenate( [ self[idx()][0].transpose(1,2,0) for _ in range(grid_size)] ,axis=1) for _ in range(grid_size) ],axis=0)
        if viz:
            fig = plt.figure(figsize=(7,7))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
        else:
            return img

    @staticmethod
    def unpickle(filename:str) -> Any:
        with open(filename,'rb') as file:
            batch = pickle.load(file,encoding='bytes')

        try:
            labels = batch[b'labels']
            data = batch[b'data'].reshape(-1,3,32,32)
            names = batch[b'filenames']
        except:
            print(type(batch))

        return data,labels,names


data_dir = Path('../data/cifar-10-batches-py/')
files = os.listdir(data_dir)

ds_train = Cifar10(data_dir,transform=lambda x: (x-x.mean())/x.std())
ds_test = Cifar10(data_dir,train=False)

split = 0.9
train_set,val_set = torch.utils.data.random_split(
    ds_train,
    ( int(len(ds_train)*(split)) , int(len(ds_train)*(1-split)) + 1 )
)

Bs = 32

train_loader = DataLoader(
    dataset=train_set,
    batch_size=Bs,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_set,
    batch_size=Bs,
    shuffle=True
)

test_loader = DataLoader(
    dataset=ds_test,
    batch_size=Bs,
    shuffle=True
)

loaders = {
    "train" : train_loader,
    "val" : val_loader,
    "test" : test_loader
}

datasets = {
    "train" : train_set,
    "val" : val_set,
    "test" : ds_test
}
