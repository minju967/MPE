from albumentations.pytorch import transforms
from torch.utils.data import Dataset
from torchvision import transforms as tf
from typing import Tuple
from torch import nn, Tensor
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import cv2
import numpy as np
import pandas as pd
import random
import torch

class fine_tuning_Dataset(Dataset):
    def __init__(self, dir, phase, opt) -> None:
        super().__init__()
        self.cls_num = opt.num_cls
        self.phase = phase
        self.image = opt.IMG
        if phase == 'Train':
            self.sample = opt.sample
        else:
            self.sample = 'False'
        self.args = opt

        self.dir = os.path.join(dir, phase)
        # if phase == 'Train':
        #     self.transforms = tf.Compose([
        #                     tf.RandomHorizontalFlip(p=0.5),
        #                     tf.RandomVerticalFlip(p=0.5),
        #                     tf.ToTensor(),
        #                     tf.Normalize
        #                     ([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        #                     ])
        # else:
        #     self.transforms = tf.Compose([
        #                     tf.ToTensor(),
        #                     tf.Normalize
        #                     ([0.485, 0.456, 0.406],
        #                      [0.229, 0.224, 0.225])
        #                     ])
        
        # self.gray = tf.Grayscale()
        if self.args.Gray == 'True':
            gray = 1
        else:
            gray = 0

        if phase == 'Train':
            self.transforms = A.Compose([
                A.Resize(height=224, width=224),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ToGray(p=gray),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(height=224, width=224),
                A.ToGray(p=gray),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std =(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        
        
        self.gray = A.ToGray()

        if self.sample == 'True':
            data_list = pd.read_csv(self.dir+'\\sampling_k200.csv')
            self.image_list = data_list.loc[:,'Name'].values.tolist()
            self.label_df = data_list

        else:
            self.image_list = [file for file in os.listdir(self.dir) if file.endswith('.png')]
            self.label_df = pd.read_csv(self.dir+'\\label.csv')


    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[Tensor]:

        image_name = self.image_list[index]
        img_path = os.path.join(f'D:\\IMAGE\\{self.image}', self.phase, image_name)
        if not os.path.exists(img_path):
            if self.phase == 'Train':
                img_path = os.path.join(f'D:\\IMAGE\\{self.image}', 'Test', image_name)
            else:
                img_path = os.path.join(f'D:\\IMAGE\\{self.image}', 'Train', image_name)
        
        image = cv2.imread(img_path)
        
        if self.cls_num == 2:
            target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][5:7]
        else:
            target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][3:8]

        target = np.array(target).astype(np.float32)
        if self.transforms is not None:
            augmentations = self.transforms(image=image)
            image = augmentations["image"]

        return image, target, image_name


