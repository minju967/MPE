from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
from torch import nn, Tensor
from PIL import Image

import os
import random
import numpy as np
import pandas as pd


class fine_tuning_Dataset(Dataset):
    def __init__(self, dir, phase) -> None:
        super().__init__()
        self.dir = os.path.join(dir, phase)
        if phase == 'Train':
            self.transforms = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.5),
                            transforms.RandomVerticalFlip(p=0.5),
                            transforms.Resize((256,256)),
                            transforms.ToTensor(),
                            transforms.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])
        else:
            self.transforms = transforms.Compose([
                            transforms.Resize((256,256)),
                            transforms.ToTensor(),
                            transforms.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])

        self.image_list = [file for file in os.listdir(self.dir) if file.endswith('.png')]
        random.shuffle(self.image_list)

        self.label_df = pd.read_csv(self.dir+'\\label.csv')

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_name = self.image_list[index]
        img_path = os.path.join(self.dir, image_name)        

        image = Image.open(img_path).convert('RGB')
        target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][3:9]

        target = np.array(target).astype(np.float32)
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target, image_name

from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from torchvision.transforms import (
                                    RandomResizedCrop,
                                    RandomRotation,
                                    RandomHorizontalFlip,
                                    ColorJitter,
                                    RandomGrayscale,
                                    RandomApply,
                                    Compose,
                                    GaussianBlur,
                                    Resize,
                                    ToTensor,
                                )

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        views = [self.base_transform(x) for i in range(self.n_views)]
        return views

class ContrastiveLearningDataset:
    def get_complete_transform(output_shape, kernel_size, s=1.0):
        """
        The color distortion transform.
        
        Args:
            s: Strength parameter.
        
        Returns:
            A color distortion transform.
        """
        bg_color = (255, 255, 255)
        # rnd_crop = RandomResizedCrop(output_shape)
        rnd_flip          = RandomHorizontalFlip(p=0.5)
        rnd_rotation      = RandomRotation(360, fill=bg_color, expand=True)
        color_jitter      = ColorJitter(hue=0.24)
        rnd_color_jitter  = RandomApply([color_jitter], p=0.8)

        rnd_gray          = RandomGrayscale(p=0.2)
        gaussian_blur     = GaussianBlur(kernel_size=kernel_size)
        rnd_gaussian_blur = RandomApply([gaussian_blur], p=0.5)

        resize            = Resize((224, 224))
        to_tensor         = ToTensor()

        image_transform   = Compose([to_tensor,
                                    rnd_rotation,
                                    # rnd_flip,
                                    # rnd_color_jitter,
                                    # rnd_gray,
                                    # rnd_gaussian_blur,
                                    resize
                                ])
        return image_transform

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
        
        if self.transform:
            image = self.transform(image)

        return image