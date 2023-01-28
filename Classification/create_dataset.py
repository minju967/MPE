from albumentations.pytorch import transforms
import albumentations
from torch.utils.data import Dataset
from torchvision import transforms as tf
from typing import Tuple
from torch import nn, Tensor
from PIL import Image

import os
import cv2
import numpy as np
import pandas as pd
import random

class fine_tuning_Dataset(Dataset):
    def __init__(self, dir, phase, num, sample) -> None:
        super().__init__()
        self.cls_num = num
        self.phase = phase
        self.dir = os.path.join(dir, phase)
        if phase == 'Train':
            self.transforms = tf.Compose([
                            tf.RandomHorizontalFlip(p=0.5),
                            tf.RandomVerticalFlip(p=0.5),
                            tf.ToTensor(),
                            tf.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])
        else:
            self.transforms = tf.Compose([
                            tf.ToTensor(),
                            tf.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])

        if sample == 'True':
            self.image_list = []
            data_list = pd.read_csv(self.dir+'\\label.csv')
            cls = ['A', 'HSM', 'Wire']
            for c in cls:
                data = data_list[data_list[c]==1]['Name'].values.tolist()
                self.image_list.extend(data)
            data = data_list[(data_list['A']==0)&(data_list['HSM']==0)&(data_list['Wire']==0)&(data_list['C']==1)&(data_list['Cut']==1)]['Name'].values.tolist()
            self.image_list.extend(random.choices(data, k=300))
            data = data_list[(data_list['A']==0)&(data_list['HSM']==0)&(data_list['Wire']==0)&(data_list['C']==0)&(data_list['Cut']==1)]['Name'].values.tolist()
            self.image_list.extend(random.choices(data, k=200))
        else:
            self.image_list = [file for file in os.listdir(self.dir) if file.endswith('.png')]

        self.label_df = pd.read_csv(self.dir+'\\label.csv')

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_name = self.image_list[index]

        # img_path = os.path.join(self.dir, image_name)        
        img_path = os.path.join('D:\\IMAGE\\PyVista_IMAGE_DATA', self.phase, image_name)
        if not os.path.exists(img_path):
            if self.phase == 'Train':
                img_path = os.path.join('D:\\IMAGE\\PyVista_IMAGE_DATA', 'Test', image_name)
            else:
                img_path = os.path.join('D:\\IMAGE\\PyVista_IMAGE_DATA', 'Train', image_name)
        
        image = Image.open(img_path).convert('RGB')
        
        if self.cls_num == 2:
            target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][5:7]
        else:
            target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][3:8]

        target = np.array(target).astype(np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
            
        return image, target, image_name


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, n_views=2):
        self.n_views = n_views

    def __call__(self, x):
        self.base_transform = albumentations.Compose([
                                albumentations.Resize(256, 256), 
                                albumentations.RandomCrop(224, 224),
                                albumentations.Rotate(limit=180, p=1, border_mode=cv2.BORDER_REPLICATE),
                                albumentations.RGBShift(p=0.3),
                                albumentations.Blur(blur_limit=(21, 21), p=0.3),
                                transforms.ToTensorV2(transpose_mask=True)     
                                ])
        views = [self.base_transform(image=x) for _ in range(self.n_views)]
        return views

class ContrastiveLearningDataset(Dataset):
    def __init__(self, root_folder, phase, transform):
        self.dir           = os.path.join(root_folder, phase)
        self.image_list    = [file for file in os.listdir(self.dir) if file.endswith('.png')]
        self.transform     = albumentations.Compose([
                            albumentations.Resize(256, 256), 
                            albumentations.RandomCrop(224, 224),
                            albumentations.Rotate(limit=180, p=1, border_mode=cv2.BORDER_REPLICATE),
                            albumentations.RGBShift(p=0.3),
                            albumentations.Blur(blur_limit=(21, 21), p=0.3),
                            transforms.ToTensorV2(transpose_mask=True)     
                             ])
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):            
        image_name = self.image_list[idx]
        img_path = os.path.join(self.dir, image_name) 
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = [self.transform(image=image) for i in range(2)]
        return image

class PyTorch_BYOL(Dataset):
    def __init__(self, root_folder, phase, transform):
        self.dir           = os.path.join(root_folder, phase)
        self.image_list    = [file for file in os.listdir(self.dir) if file.endswith('.png')]
        self.transform     = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):            
        image_name = self.image_list[idx]
        img_path = os.path.join(self.dir, image_name) 
        
        image = Image.open(img_path).convert('RGB')
        image = [self.transform(image) for _ in range(2)]

        return image