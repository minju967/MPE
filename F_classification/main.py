import argparse
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import torch
from create_dataset import (ContrastiveLearningDataset,
                            ContrastiveLearningViewGenerator,
                            fine_tuning_Dataset)
from Fine_tuning.models import ResNet18, finetuning_Model
from Fine_tuning.train import cls_Trainer
from SimCLR_SSL.models import ResNetSimCLR
from SimCLR_SSL.simclr import SimCLR
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

    os.makedirs(os.path.join(args.save, f'experiment_{idx}'))

    return f'experiment_{idx}'

def view_data(dataset, index):
    for i in range(1,6):       
        images = dataset[index]
        view1, view2 = images[0]['image'], images[1]['image']
        plt.subplot(5,2,2*i-1)
        plt.imshow(view1.permute(1,2,0))
        plt.subplot(5,2,2*i)
        plt.imshow(view2.permute(1,2,0))

    plt.show()

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

    # Pre-train model Fine tuning
    if args.pretrain == 0 or args.pretrain == 2:
        stage = {0:"IMAGENET Pre-trained Model", 2:"SimCLR Pre-trained Model"}
        str = f'__Start Model Fine-Tuning: " {stage[args.pretrain]} " __Backbone:{args.backbone}'
        
        print(str)
        print('='*len(str))

        train_dataset   = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Train')
        test_dataset    = fine_tuning_Dataset(os.path.join(args.Root, args.data), phase='Test')

        train_loader    = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
        test_loader     = DataLoader(test_dataset,  batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        if args.pretrain == 0:
            pt_path         = False
            model           = finetuning_Model(model_name=args.backbone, model_PT=pt_path, cls_num=args.num_cls).to(device)
        else:
            if args.SSR_SimCLR:
                pt_path         = os.path.join('D:\\VS_CODE\\Paper\\EX_Result2', args.pt_path, 'checkpoint_best.pth.tar')
                model           = finetuning_Model(model_name=args.backbone, model_PT=pt_path, cls_num=args.num_cls).to(device)
            else:
                pt_path         = os.path.join('D:\\VS_CODE\\Paper\\Result', args.pt_path, 'model.pth')
                model           = ResNet18(model_name=args.backbone, model_PT=pt_path, cls_num=args.num_cls).to(device)

        summary_(model,(3,256,256),batch_size=args.batch_size)

        with torch.cuda.device(args.gpu_index):
            Trainer         = cls_Trainer(model=model, train_dataset=train_loader, test_data=test_loader, args=args, device=device, save=save_path, loss_weight=args.weight)
            Trainer.train(args.epochs)
        
    # SimCLR Representation learning 
    elif args.pretrain == 1:
        if args.pre_RL:
            str = f'__Start Model Pre-train: " Representation Learning "__Backbone: Pre-trained {args.backbone}'
        else:
            str = f'__Start Model Pre-train: " Representation Learning "__Backbone: {args.backbone}'
        f.write(str)
        print(str)
        print('='*len(str))
        # dataset: https://www.kaggle.com/code/aritrag/simclr

        # Show dataset
        custom_transform = ContrastiveLearningViewGenerator(n_views=2)
        train_dataset    = ContrastiveLearningDataset(os.path.join(args.Root, args.data), phase='Train', transform=custom_transform)

        if args.show:
            plt.figure(figsize=(10,20))
            view_data(train_dataset, 1)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=args.workers, pin_memory=True, drop_last=True)
        f.close()
        
        model       = ResNetSimCLR(base_model=args.backbone, out_dim=args.out_dim, pretrain=args.pre_RL)
        optimizer   = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)


        #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args, save=save_path, device=device)
            simclr.train(train_loader)

    return 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Root', type=str, default='D:\\VS_CODE\\Paper\\Datafolder', help="folder_name")                   
    parser.add_argument('--save', type=str, default='D:\\VS_CODE\\Paper\\F_classification\\Result', help='실험 결과 저장 폴더')   
    parser.add_argument('--pt_path', type=str, default='experiment_0', help='model_state_dict_file')   
    parser.add_argument('--pretrain', type=int, default=0, help='0: Fine-tuning model, 1: Pre-train moedl, 2: Pre-train+fine_tuning, 3: BYOL')   
    parser.add_argument('--pretrain_RL', type=bool, default=False, help="pretrained Representation learing or Not")     
    parser.add_argument('--data', type=str, default='dataset_1', help="Dataset folder")    
    parser.add_argument('--SSR_SimCLR', type=bool, default=False, help="representation Learning SimCLR or BYOL")     
    parser.add_argument('--batch_size', type=int, default=128, help="batch Size")          
    parser.add_argument('--epochs', type=int, default=10, help="Epochs")          
    parser.add_argument('--weight', type=bool, default=False, help="Add loss function weight")          
    parser.add_argument('--workers', type=int, default=8, help="workers")          
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    parser.add_argument('--show', type=bool, default=False, help="dataset show or not")          
    parser.add_argument('--backbone', type=str, default='Resnet18', choices=['Resnet18', 'Resnet34', 'Resnet50', 'EfficientNetB0', 'EfficientNetB1'], help="backbone model")    
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--num_cls', default=5, type=int, help='Number of class.')
    parser.add_argument('--n_views', default=2, type=int, help='생성 view 갯수')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--log-every-n-steps', default=10, type=int, help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')

    args = parser.parse_args()
    main(args)
