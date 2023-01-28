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

def check_exp(args):
    idx_list = []
    if os.path.exists(args.save):
        exp_list = os.listdir(args.save)
        if len(exp_list) > 0:
            for exp in exp_list:
                if 'experiment' in exp:
                    idx_list.append(int(exp.split('_')[-1]))
            idx_list.sort()
            idx = idx_list[-1] + 1
        else:
            idx = 0
    else:
        os.makedirs(args.save, exist_ok=True)
        idx = 0

    os.makedirs(os.path.join(args.save, f'experiment_{idx}'))

    return f'experiment_{idx}'

def main(args):
    experiment      = check_exp(args)
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_path       = os.path.join(args.save, experiment)

    # parameter & 실험 정보 저장
    f = open(save_path+'\\Experiment_Information.txt','w')
    f.write(f'Dataset   : {args.data}\n')
    f.write(f'Batch_size: {args.batch_size}\n')
    f.write(f'Epoch     : {args.epochs}\n')
    f.write(f'Backbone  : {args.backbone}\n')
    f.write(f'Pretrain  : {args.pretrain}\n')
    f.write(f'Experiment Explain : \n')

    stage = {0:"IMAGENET Pre-trained Model", 2:"SimCLR Pre-trained Model"}
    str = f'__Start Model Fine-Tuning: " {stage[args.pretrain]} " __Backbone:{args.backbone}'
    print(str)
    print('='*len(str))
    f.close()

    train_dataset   = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Train', num=args.num_cls,  sample=args.sample)
    test_dataset    = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Test',  num=args.num_cls,  sample='False')


    train_loader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    test_loader     = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

    if args.pt_path:
        pt_path = os.path.join(args.save, args.pt_path, 'Multi-label_best_f1_score.pt')
    else:
        pt_path = None

    print(args.PT)
    model = finetuning_Model(model_name=args.backbone, model_PT=pt_path, cls_num=args.num_cls, pt=args.PT).to(device)
    summary_(model,(3,256,256),batch_size=args.batch_size)

    with torch.cuda.device(args.gpu_index):
        Trainer = cls_Trainer(model=model, train_dataset=train_loader, test_data=test_loader, args=args, device=device, save=save_path, loss_weight=args.weight)
        Trainer.train(args.epochs)

    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="folder_name")                   
    parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\Classification\\Result', help='실험 결과 저장 폴더')   
    parser.add_argument('--pt_path', type=str, default=None, help='model_state_dict_file')   
    parser.add_argument('--pretrain', type=int, default=0, help='0: Fine-tuning model, 1: Pre-train moedl, 2: Pre-train+fine_tuning, 3: BYOL')   
    parser.add_argument('--data', type=str, default='dataset_1', help="Dataset folder")    
    parser.add_argument('--sample', type=str, default='False', help="Dataset folder")    
    parser.add_argument('--batch_size', type=int, default=128, help="batch Size")          
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")          
    parser.add_argument('--weight', type=str, default='False', help="Add loss function weight")          
    parser.add_argument('--workers', type=int, default=8, help="workers")          
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--PT', type=str, default='False', help="")                
    parser.add_argument('--backbone', type=str, default='Resnet18', choices=['Resnet18', 'Resnet34', 'Resnet50', 'EfficientNetB0', 'EfficientNetB1'], help="backbone model")    
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
