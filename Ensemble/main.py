import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_s
import numpy as np
import time
import copy
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.utils.data import DataLoader
from dataset import Ensemble_Dataset
from Ensemble.models import Model_Ensemble
from transformers import get_cosine_schedule_with_warmup

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def check_exp(args):
    save = os.path.join(args.Root, args.exp)
    idx_list = []
    if os.path.exists(save):
        exp_list = os.listdir(save)
        if len(exp_list) > 0:
            for exp in exp_list:
                if 'experiment' in exp:
                    idx_list.append(int(exp.split('_')[-1]))
            idx_list.sort()
            idx = idx_list[-1] + 1
        else:
            idx = 0
    else:
        os.makedirs(save, exist_ok=True)
        idx = 0

    os.makedirs(os.path.join(save, f'experiment_{idx}'))

    return os.path.join(save, f'experiment_{idx}')

def main(args):
    print()
    print()
    save = check_exp(args)
    # parser.add_argument('--save', type=str, default=save, help="save_path")
    args   = parser.parse_args()    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_folder = os.path.join('D:\\VS_CODE\\Paper\\Datafolder', args.data)

    ## Model Load
    stage1_path = 'D:\\VS_CODE\\Paper\\PT_File\\1stage_resnet34_\\Multi-label_best_f1_score.pt'     # 5-class classification model
    stage2_path = 'D:\\VS_CODE\\Paper\\Ensemble\\YOLO\\Runs\\train\\exp2\\weights\\best.pt'
    stage3_path = 'D:\\VS_CODE\\Paper\\PT_File\\3stage_resnet34_\\Multi-label_best_f1_score.pt'     # 2-class classification model

    # 3 network output ensemble vector 출력 
    Fix_Model = Model_Ensemble(pt_path1=stage1_path, pt_path2=stage2_path, pt_path3=stage3_path)
    Model = nn.Sequential(nn.Linear(10, args.dim), nn.ReLU(), nn.Linear(args.dim, 5)).cuda()

    # Ensemble Model Dataset Load --> resnet image, yolo image 생
    train_dataset = Ensemble_Dataset(data_folder, phase='Train')
    test_dataset  = Ensemble_Dataset(data_folder, phase='Test')

    train_loader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    test_loader     = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(Model.parameters(), lr=args.lr, eps = 1e-8)
    scheduler = lr_s.ExponentialLR(optimizer, gamma=0.95)

    min_loss      = 0
    max_recall    = 0
    max_precision = 0
    max_f1        = 0
    test_frame    = pd.DataFrame(index=range(0,0), columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

    for epoch in range(args.epochs):
        print(f'== {epoch} epoch Train Start==')
        Model.train()
        start = time.time()
        for img1, img2, target, path in train_loader:
            optimizer.zero_grad()
            images1 = img1.float().to(device)
            images2 = img2.float().to(device)

            target = target.float().to(device)
            with torch.no_grad():
                Ensemble_vec = Fix_Model(images1, images2, path)
            output = Model(Ensemble_vec)
            loss   = criterion(output, target)

            loss.backward()
            optimizer.step()
        
        if args.epochs >= 50:
            scheduler.step()

        print(f'== {epoch} epoch Train End ==')

        test_loss, test_recall, test_pre, test_f1, result = test(epoch=epoch, test_data=test_loader, model1=Fix_Model, model2=Model)
        test_frame = pd.concat([test_frame, result], axis=0)

        if (epoch > 0) and (args.epochs<50):
            optimizer.param_groups[0]["lr"] *= 0.1
        
        if (epoch+1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Learing Rate: {lr}")

        if test_loss < min_loss:
            min_loss = test_loss
            save_model(model=Model, phase='Loss', save=save)
        if test_recall > max_recall:
            max_recall = test_recall
            save_model(model=Model, phase='Recall', save=save)
        if test_pre > max_precision:
            max_precision = test_pre
            save_model(model=Model, phase='Precision', save=save)
        if test_f1 > max_f1:
            max_f1 = test_f1
            save_model(model=Model, phase='F1', save=save)
        
        print()
        learning_time = time.time()-start
        print(f'{epoch} Epoch Time: {learning_time//60}m {learning_time%60:.3f}s')

    test_frame.to_csv(os.path.join(save, 'test.csv'))

def save_model(model, phase, save):
    name = {'Loss':'Multi-label_best_Loss.pt', 'Precision':'Multi-label_best_precision.pt',
            'Recall':'Multi-label_best_recall.pt','F1':'Multi-label_best_f1_score.pt'}

    pt_path = os.path.join(save, name[phase])
    torch.save({'model_state_dict': model.state_dict()}, pt_path)

def Average( lst):
    return sum(lst) / len(lst)

def cls_matrix(epoch, Predicts, Labels):
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

        Acc = accuracy_score(predict, target)
        Precision = precision_score(target, predict, zero_division=0)
        Recall = recall_score(target, predict, zero_division=0)
        F1_score = f1_score(target, predict, zero_division=0)

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
    print(f'{epoch} Epoch Test Accuracy:{acc:.3f}\t Precision:{precision:.3f}\t Recall:{recall:.3f}\t F1_score:{f1:.3f}')
    print()

    data = [round(i, 3) for i in [epoch, acc, precision, recall, f1]]
    data = pd.DataFrame(data=[data], columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

    return recall, precision, f1, data


@torch.no_grad()
def test(epoch, test_data, model1, model2):
    model2.eval()
    criterion = nn.MultiLabelSoftMarginLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Predicts = torch.rand(size=(0, args.num_cls)).to(device)
    Labels   = torch.rand(size=(0, args.num_cls)).to(device)
    test_loss = 0
    iter = 0

    for img1, img2, targets, path in test_data:
        images1 = img1.float().to(device)
        images2 = img2.float().to(device)

        targets = targets.float().to(device)
        Ensemble_vec = model1(images1, images2, path)

        outputs = model2(Ensemble_vec)
        test_loss += criterion(outputs, targets).item()

        outputs = outputs > 0.5            
        Predicts = torch.cat([Predicts, outputs.long()], dim=0)
        Labels   = torch.cat([Labels, targets], dim=0)

    test_recall, test_pre, test_f1, result = cls_matrix(epoch, Predicts, Labels)
    
    return test_loss, test_recall, test_pre, test_f1, result        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()    
    parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Ensemble', help="folder_name")                   
    parser.add_argument('--data', type=str, default='dataset_1', help="Dataset folder") 
    parser.add_argument('--exp', type=str, default='Dimension', help="Experiment factor")  
    parser.add_argument('--IMG', type=str, default='Python_IMAGE_DEPTH_COLOR', help="Image type")      
    parser.add_argument('--batch_size', type=int, default=256, help="batch Size")          
    parser.add_argument('--epochs', type=int, default=30, help="Epochs")   
    parser.add_argument('--num_cls', type=int, default=5, help="Epochs")    
    parser.add_argument('--dim', type=int, default=10, help="Epochs")                              
    parser.add_argument('--workers', type=int, default=0, help="workers")          
    parser.add_argument('--backbone', type=str, default='Resnet34', choices=['Resnet18', 'Resnet34', 'Resnet50', 'EfficientNetB0', 'EfficientNetB1'], help="backbone model")    
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--PT', type=str, default='False', help="")                

    args = parser.parse_args()
    main(args)
