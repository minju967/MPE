from albumentations.pytorch import transforms
import albumentations
from torch.utils.data import Dataset
from torchvision import transforms as tf
from typing import Tuple
from torch import nn, Tensor
from PIL import Image

import time
import os
import cv2
import numpy as np
import pandas as pd
import random
import glob

from pathlib import Path
from YOLO.utils.augmentations import letterbox

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes

class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=416, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True)))  # glob
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  # dir
            elif os.path.isfile(p):
                files.append(p)  # files
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        ni, nv = len(images), 0

        self.img_size = img_size
        self.stride = stride
        self.files = images 
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = tf.Compose([
                            tf.Resize((416, 416)),
                            # tf.RandomHorizontalFlip(p=0.5),
                            # tf.RandomVerticalFlip(p=0.5),
                            tf.ToTensor(),
                            tf.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])    # optional
        self.vid_stride = vid_stride  # video frame-rate stride
        self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        # Read image
        self.count += 1
        im0 = Image.open(path).convert('RGB')
        assert im0 is not None, f'Image Not Found {path}'
        s = f'image {self.count}/{self.nf} {path}: '

        if self.transforms:
            im = self.transforms(im0)  # transforms
        else:
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  # padded resize
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            im = np.ascontiguousarray(im)  # contiguous

        return im

class Ensemble_Dataset(Dataset):
    def __init__(self, path, phase) -> None:
        super().__init__()
        self.phase = phase
        self.path = os.path.join(path, phase)   # dataset folder
        self.transform1_train = tf.Compose([
                            tf.RandomHorizontalFlip(p=0.5),
                            tf.RandomVerticalFlip(p=0.5),
                            tf.ToTensor(),
                            tf.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])            # resnet transform
        self.transform2_train = None            # yolov5 transform
        self.transform1_test = tf.Compose([
                            tf.ToTensor(),
                            tf.Normalize
                            ([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
                            ])                  # resnet transform
        self.transform2_test = None             # yolov5 transform

        self.image_list = [file for file in os.listdir(self.path) if file.endswith('.png')]
        self.label_df = pd.read_csv(self.path+'\\label.csv')

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> Tuple[Tensor]:
        image_name = self.image_list[index]
        img_path = os.path.join(self.path, image_name)

        image = Image.open(img_path).convert('RGB')     # 이미지

        target = self.label_df[self.label_df.Name==image_name].values.tolist()[0][3:8]
        target = np.array(target).astype(np.float32)    # 정답

        start = time.time()
        if self.phase == 'Train':
            if self.transform1_train is not None:
                image1 = self.transform1_train(image)
        else:
            image1 = self.transform1_test(image)

        dataset = LoadImages(path=img_path, img_size=416, stride=32, auto=True, vid_stride=1)

        for img in dataset:
            image2 = img

        return image1, image2, target, img_path
