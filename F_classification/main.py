import os
import csv
import cv2
import torch
import torch.nn
import numpy as np
import pandas as pd
import argparse
import torch.optim as optim
import glob
import random
from time import time
from torchsummary import summary
from PIL import Image
from typing import Tuple
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from IPython.display import display
from torch.nn import functional as F

from torchvision import transforms
from torchvision.models import resnet50
from efficientnet_pytorch import EfficientNet


class Dataset(Dataset):
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

class Model(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name == 'Resnet50':
            self.model = resnet50(pretrained=True)
            self.classifier = nn.Linear(1000, 6)

        elif model_name == 'EfficientNet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
            self.classifier = nn.Linear(1000, 6)

    def forward(self, x):
        if self.model_name == 'Resnet50':
            x = self.model(x)
        elif self.model_name == 'EfficientNet':
            x = F.relu(self.model(x))

        x = self.classifier(x)
        return x

def check_exp(args):
    idx_list = []
    exp_list = os.listdir(args.save)
    if len(exp_list) > 0:
        for exp in exp_list:
            if 'experiment' in exp:
                idx_list.append(int(exp.split('_')[-1]))
        idx_list.sort()
        idx = idx_list[-1] + 1
    else:
        idx = 0

    os.makedirs(os.path.join(args.save, f'experiment_{idx}'))

    return f'experiment_{idx}'

def main(args, Exp):
    train_dataset   = Dataset(os.path.join(args.Root, args.data), phase='Train')
    test_dataset    = Dataset(os.path.join(args.Root, args.data), phase='Test')

    train_loader    = DataLoader(train_dataset, batch_size=128, num_workers=0, shuffle=True)
    test_loader     = DataLoader(test_dataset, batch_size=128, num_workers=0, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = Model(model_name=args.backbone).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MultiLabelSoftMarginLoss()

    num_epoch = 10

    max_acc = 0
    save_model_path = os.path.join(args.save, Exp)

    for epoch in range(num_epoch):
        model.train()
        Start_time = time()
        print('='*30)
        print(f'Epoch:{epoch}__Start')
        for i, (images, targets, _) in enumerate(train_loader):
            optimizer.zero_grad()

            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()
    
            if (i+1) % 5 == 0:
                outputs = outputs > 0.5
                acc = (outputs == targets).float().mean()
                print(f'[{i}/{len(train_loader)}]: Loss:{loss.item():.5f}, Acc:{acc.item():.5f}')
            
        End_time = time()
        Learning_time = End_time - Start_time
        m = Learning_time // 60
        s = Learning_time % 60
        print(f'\nLearning Time: {int(m)}min. {int(m)}sec.')

        model.eval()
        batch_size = test_loader.batch_size
        batch_index = 0
        test_acc = 0.0        
        test_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path'])

        for i, (images, targets, name) in enumerate(test_loader):
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            test_acc += acc
            outputs = outputs.long().squeeze(0).detach().cpu().numpy()
            batch_index = i * batch_size
            
            idx  = list(range(batch_index,batch_index+images.shape[0]))
            path = [os.path.join(args.Root, args.data, img) for img in name]
            A    = list(outputs[:,0])
            HSM  = list(outputs[:,1])
            C    = list(outputs[:,2])
            Cut  = list(outputs[:,3])
            Wire = list(outputs[:,4])
            none = list(outputs[:,5])

            df = pd.DataFrame({'Idx':idx, 'Name':name, 'A':A, 'HSM':HSM, 'C':C, 'Cut':Cut, 'Wire':Wire, 'None':none, 'Path':path})
            test_df = pd.concat([test_df, df], ignore_index=True)

        print(f'Test_data acc: {test_acc/len(test_loader):.5f}\n')
        
        # Model 저장
        if max_acc < test_acc:
            torch.save(model.state_dict(), os.path.join(save_model_path, 'model_state_dict.pt'))   # 모델 객체의 state_dict 저장
            test_df.to_csv(os.path.join(args.save, Exp, f'result_{epoch}.csv'))
            max_acc = test_acc

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="folder_name")                   
    parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\Result', help='실험 결과 저장 폴더')   
    parser.add_argument('--data', type=str, default='dataset_0', help="Save folder")          
    parser.add_argument('--backbone', type=str, default='EfficientNet', help="backbone model")          
    args = parser.parse_args()  

    exp = check_exp(args)
    main(args, exp)
