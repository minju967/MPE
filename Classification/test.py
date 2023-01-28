import argparse
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import torch
from create_dataset import (ContrastiveLearningDataset,
                            ContrastiveLearningViewGenerator,
                            fine_tuning_Dataset)
from Fine_tuning.models import finetuning_Model
from Fine_tuning.train import cls_Trainer
from torch.utils.data import DataLoader
from torchsummary import summary as summary_

from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def visualize_confuse_matrix(matrix, path):
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]

    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=20) # Plot 이름
    # plt.tight_layout()
    # plt.colorbar()
    label=["C", "CUT"] # 라벨값
    tick_marks = np.arange(len(label)) 
    plt.xticks(tick_marks, label)
    plt.yticks(tick_marks, label)
    plt.xlabel('Predicted', fontsize=15)
    plt.ylabel('True', fontsize=15)
    # 표 안에 숫자 기입하는 방법
    name = [['TP','FN'], ['FP', 'TN']]
    thresh = matrix.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix[i, j]),
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=16)
        
    # plt.show()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)


def Average(lst):
    return sum(lst) / len(lst)

def cls_matrix(args, Predicts, Labels):
    if args.num_cls == 2:
        cls_list = {0:'C', 1:'Cut'}
    else:
        cls_list = {0:'A', 1:'HSM', 2:'C', 3:'Cut', 4:'Wire'}

    acc         = []
    precision   = []
    recall      = []
    f1          = []

    for i in range(args.num_cls):
        predict = Predicts[:,i].detach().cpu().tolist()
        target = Labels[:,i].detach().cpu().tolist()

        Acc = accuracy_score(target, predict)
        Precision = precision_score(target, predict)
        Recall = recall_score(target, predict)
        F1_score = f1_score(target, predict)

        acc.append(Acc)
        precision.append(Precision)
        recall.append(Recall)
        f1.append(F1_score)

        print(f'{cls_list[i]} Matrix: Accuracy:{Acc:.3f}\t Precision:{Precision:.3f}\t Recall:{Recall:.3f}\t F1_score:{F1_score:.3f}')
    
    acc = Average(acc)
    precision = Average(precision)
    recall = Average(recall)
    f1 = Average(f1)
    print()
    print(f'Test Accuracy:{acc:.3f}\t Precision:{precision:.3f}\t Recall:{recall:.3f}\t F1_score:{f1:.3f}')
    print()
    return

@torch.no_grad()
def main(args):
    save_path = os.path.join('D:\VS_CODE\Paper\Classification', 'test')
    os.makedirs(save_path, exist_ok=True)

    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset    = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Test',  num=args.num_cls, sample='False')
    test_loader     = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    pt_path = 'D:\\VS_CODE\\Paper\\Classification\\Label5_clssification\\experiment_3\\Multi-label_best_f1_score.pt'
    model = finetuning_Model(model_name='Resnet34', model_PT=pt_path, cls_num=args.num_cls, pt='False').to(device)

    model.eval()

    Predicts = torch.rand(size=(0, args.num_cls))
    Labels   = torch.rand(size=(0, args.num_cls))

    for iter, (images, targets, _) in enumerate(test_loader):
        images  = images.to(device)
        targets = targets.to(device)

        outputs = model(images)

        outputs = outputs > 0.5            
        Predicts = torch.cat([Predicts, outputs.detach().cpu().long()], dim=0)
        Labels   = torch.cat([Labels, targets.detach().cpu().long()], dim=0)

    cls_matrix(args, Predicts, Labels)



    # Predicts = Predicts.flatten()
    # Labels   = Labels.flatten()
    # cf_matrix = confusion_matrix(Labels, Predicts)

    
    # visualize_confuse_matrix(cf_matrix, os.path.join(save_path, 'figure5.svg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="folder_name")                   
    parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\Classification\\Result', help='실험 결과 저장 폴더')   
    parser.add_argument('--pt_path', type=str, default=None, help='model_state_dict_file')   
    parser.add_argument('--pretrain', type=int, default=0, help='0: Fine-tuning model, 1: Pre-train moedl, 2: Pre-train+fine_tuning, 3: BYOL')   
    parser.add_argument('--data', type=str, default='dataset_1', help="Dataset folder")    
    parser.add_argument('--batch_size', type=int, default=128, help="batch Size")          
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")          
    parser.add_argument('--weight', type=str, default='False', help="Add loss function weight")          
    parser.add_argument('--workers', type=int, default=8, help="workers")          
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--PT', type=str, default='False', help="")                
    parser.add_argument('--backbone', type=str, default='Resnet34', choices=['Resnet18', 'Resnet34', 'Resnet50', 'EfficientNetB0', 'EfficientNetB1'], help="backbone model")    
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--num_cls', default=5, type=int, help='Number of class.')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--log-every-n-steps', default=10, type=int, help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')

    args = parser.parse_args()
    main(args)