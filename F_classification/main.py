import os
import cv2
import csv
import torch
import torch.nn
import glob
import random
import logging
import argparse
import numpy             as np
import pandas            as pd
import torch.optim       as optim
import matplotlib.pyplot as plt

from time   import time
from PIL    import Image
from typing import Tuple
from torch  import nn, Tensor

from torchsummary           import summary
from torchvision            import transforms
from torch.nn               import functional as F
from IPython.display        import display
from logging.config         import dictConfig
from efficientnet_pytorch   import EfficientNet
from torch.utils.data       import Dataset, DataLoader
from torchvision.models     import resnet50, resnet18, resnet34

from Representation.simclr  import SimCLR
from Representation.models  import ResNetSimCLR
from Fine_tuning.models     import finetuning_Model
from create_dataset         import ContrastiveLearningViewGenerator
from create_dataset         import fine_tuning_Dataset, ContrastiveLearningDataset
from create_dataset         import ContrastiveLearningDataset as contrastive_dataset

# logger = logging.getLogger()

parser = argparse.ArgumentParser()
parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="folder_name")                   
parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\Result', help='실험 결과 저장 폴더')   
parser.add_argument('--data', type=str, default='dataset_0', help="Save folder")     
parser.add_argument('--batch_size', type=int, default=256, help="batch Size")          
parser.add_argument('--epochs', type=int, default=10, help="Epochs")          
parser.add_argument('--workers', type=int, default=0, help="workers")          
parser.add_argument('--show', type=bool, default=False, help="dataset show or not")          
parser.add_argument('--backbone', type=str, default='Resnet18', help="backbone model")    
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

class cls_Trainer():
    def __init__(self, model, train_dataset, test_data, args, device, save) -> None:
        super().__init__()
        self.model = model
        self.train_data = train_dataset
        self.test_data = test_data
        self.device = device
        self.args =  args
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.save_model_path = save
        self.test_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path'])

        dictConfig({
        'version': 1,
        'formatters': {
            'default': {
                'format':'[%(asctime)s] %(message)s'
            }
        },
        'handlers': {
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': os.path.join(save, 'debug.log'),
                'formatter': 'default',
            },
        },
        'root': {
            'level': 'DEBUG',
            'handlers': ['file']
        }
        })  

    def train(self, epochs):
        max_acc = 0

        for epoch in range(epochs):
            Start_time = time()
            print('='*30)
            print(f'Epoch:{epoch}__Start')

            self.model.train()
            
            for i, (images, targets, _ ) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
                outputs = outputs > 0.5
                acc = (outputs == targets).float().mean()
                logging.info(f'[Train][{i}|{len(self.train_data)}] Loss:{loss.item():.5f} Acc:{acc.item():.5f}')

                if (i+1) % 5 == 0:
                    print(f'[{i}/{len(self.train_data)}]: Loss:{loss.item():.5f}, Acc:{acc.item():.5f}')
                
            End_time = time()
            Learning_time = End_time - Start_time
            m = Learning_time // 60
            s = Learning_time % 60
            print(f'\nLearning Time: {int(m)}min. {int(s)}sec.')

            test_acc = self.test(epoch=epoch, args=self.args)

            if test_acc > max_acc:
                torch.save(self.model.state_dict(), os.path.join(self.save_model_path, 'model_state_dict.pt'))   # 모델 객체의 state_dict 저장
                self.test_df.to_csv(os.path.join(self.save_model_path, f'result_{epoch}.csv'))
                max_acc = test_acc
        return

    def test(self, epoch, args):
        self.model.eval()
        batch_size = self.test_data.batch_size
        batch_index = 0
        test_acc = 0.0  
        self.test_df = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path'])

        total_test = 0
        num_acc = 0

        for i, (images, targets, name) in enumerate(self.test_data):
            images = images.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(images)

            total_test += images.shape[0]

            outputs = outputs > 0.5
            acc = (outputs == targets).float().mean()
            test_acc += acc
            outputs = outputs.long().squeeze(0).detach().cpu().numpy()
            
            batch_index = i * batch_size
            
            idx  = list(range(batch_index,batch_index+images.shape[0]))
            path = [os.path.join(args.Root, args.data, 'Test', img) for img in name]
            A    = list(outputs[:,0])
            HSM  = list(outputs[:,1])
            C    = list(outputs[:,2])
            Cut  = list(outputs[:,3])
            Wire = list(outputs[:,4])
            none = list(outputs[:,5])

            targets = targets.long().detach().cpu().numpy()
            results = []
            for i in range(images.shape[0]):
                if list(outputs[i]) == list(targets[i]):
                    results.append(True)
                else:
                    results.append(False)

            df = pd.DataFrame({'Idx':idx, 'Name':name, 'A':A, 'HSM':HSM, 'C':C, 'Cut':Cut, 'Wire':Wire, 'None':none, 'Path':path, 'Result':results})
            self.test_df = pd.concat([self.test_df, df], ignore_index=True)

            num_acc += sum(list(map(int, results)))

        epoch_test_acc = num_acc/total_test
        logging.info(f'[Test][{epoch}] Acc:{epoch_test_acc:.5f}')
        print(f'Test_data Acc: {num_acc/total_test:.5f}\n')
        
        return epoch_test_acc

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

def view_data(dataset, index):
    for i in range(1,6):       
        images = dataset[index]
        view1, view2 = images
        plt.subplot(5,2,2*i-1)
        plt.imshow(view1.permute(1,2,0))
        plt.subplot(5,2,2*i)
        plt.imshow(view2.permute(1,2,0))
    plt.show()

def main():
    args            = parser.parse_args()
    experiment      = check_exp(args)

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path       = os.path.join(args.save, experiment)

    # Pre-train model Fine tuning
    '''
    train_dataset   = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Train')
    test_dataset    = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Test')

    train_loader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
    test_loader     = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

    model           = finetuning_Model(model_name=args.backbone).to(device)
    print(model)
    Trainer         = cls_Trainer(model=model, train_dataset=train_loader, test_data=test_loader, args=args, device=device, save=save_path)
    Trainer.train(args.epochs)
    '''
    
    # Representation learning 
    # dataset: https://www.kaggle.com/code/aritrag/simclr
    # Show dataset
    output_shape     = [224,224]
    kernel_size      = [21,21] # 10% of the output_shape
    base_transforms  = contrastive_dataset.get_complete_transform(output_shape=output_shape, kernel_size=kernel_size, s=1.0)
    custom_transform = ContrastiveLearningViewGenerator(base_transform=base_transforms, n_views=2)
    train_dataset    = ContrastiveLearningDataset(os.path.join(args.Root, args.data), phase='Train', transform=custom_transform)

    if args.show:
        plt.figure(figsize=(10,20))
        view_data(train_dataset, 2000)

    train_loader = torch.utils.data.DataLoader(
                            train_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, pin_memory=True, drop_last=True)

    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    with torch.cuda.device(args.gpu_index):
        simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        simclr.train(train_loader)

    return 


if __name__ == '__main__':
    main()
