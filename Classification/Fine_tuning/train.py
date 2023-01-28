from torch.utils.tensorboard import SummaryWriter
from time import time

import os
import copy
import torch
import torchvision
import torch.nn          as nn
import logging
import pandas            as pd
import torch.optim       as optim

from Fine_tuning.pytorchtools import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

os.environ['KMP_DUPLICATE_LIB_OK']='True'

torch.manual_seed(0)
logger = logging.getLogger(__name__)

class cls_Trainer():
    def __init__(self, model, train_dataset, test_data, args, device, save, loss_weight) -> None:
        super().__init__()
        self.model           = model
        self.train_data      = train_dataset
        self.test_data       = test_data
        self.device          = device
        self.args            = args
        self.save_model_path = save
        self.max_f1          = 0
        
        if args.PT:
            self.optimizer       = optim.Adam(model.parameters(), lr=1e-4)
        else:
            self.optimizer       = optim.Adam(model.parameters(), lr=1e-3)
        self.writer          = SummaryWriter(self.save_model_path)
        # self.early_stopping  = EarlyStopping(patience = 5, verbose=True)

        if loss_weight == 'False':
            print(f'No weight Loss Function')
            self.criterion = nn.MultiLabelSoftMarginLoss()
        else:
            print(f'With weight Loss Function')
            nSamples     = self.cls_sample()
            loss_Weights = [1 - (x / sum(nSamples)) for x in nSamples]
            loss_Weights = torch.FloatTensor(loss_Weights).to(device)
            self.criterion = nn.MultiLabelSoftMarginLoss(weight=loss_Weights, reduction='mean')

        # train/test matrix 결과 저장
        self.test_frame     = pd.DataFrame(index=range(0,0), columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1_score'])

        dataiter = iter(self.train_data)
        images, _, _ = dataiter.next()

        img_gird = torchvision.utils.make_grid(images[:6,:,:,:])
        self.writer.add_image('Train_images_Samples', img_gird)

        self.Add_txt()

        logging.basicConfig(filename=os.path.join(self.save_model_path, 'training.log'), level=logging.DEBUG)
        logging.getLogger('matplotlib.font_manager').disabled = True

    def Add_txt(self):
        Represnetaion = {0:'Fine-tuning model', 1:'Pre-train moedl', 2:'Pre-train+fine_tuning', 3:'BYOL'}
        self.writer.add_text('Parameters', f' Representaion:{Represnetaion[self.args.pretrain]}')
        self.writer.add_text('Parameters', f' Backbone:{self.args.backbone}')
        self.writer.add_text('Parameters', f' Loss weights adaptaion:{self.args.weight}')


    def cls_sample(self):
        csv_path = os.path.join(self.args.Root, self.args.data, 'Train', 'label.csv')
        data_list = pd.read_csv(csv_path)
        samples = []
        if self.args.num_cls == 2:
            cls_list = ['C', 'Cut']    
        else:
            cls_list = ['A', 'HSM', 'C', 'Cut', 'Wire']

        for cls in cls_list:
            datas = data_list.loc[data_list[cls]==1]
            samples.append(len(datas.index))

        print(f'Number of Train class data: {samples}\n')
        return samples

    def train(self, epochs):
        #######################################
        #   Multi-label classifier Training
        #######################################
        min_loss      = 1000
        max_recall    = 0
        max_precision = 0
        # max_f1        = 0

        for epoch in range(epochs):
            print(f'{epoch} Epoch Train Start')
            self.model.train()
            for iter, (images, targets, name) in enumerate(self.train_data):
                self.optimizer.zero_grad()
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
        
                self.writer.add_scalar('Train_loss', loss.item(), global_step=iter)
            #===================================  Epoch Finish  ===================================
            # 모델 Test
            test_loss, test_recall, test_pre, test_f1 = self.test(epoch=epoch)
            logging.info(f'\n\n____ Finish Fine-Tuning: [Loss] {test_loss:.3f}\t[Recall] {test_recall:.3f}\t[Precision] {test_pre:.3f}\t[F1_score] {test_f1:.3f}____')

            if test_loss < min_loss:
                torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_Loss.pt'))   # 모델 객체의 state_dict 저장
                min_loss = test_loss    
            if test_recall > max_recall:
                torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_recall.pt'))   # 모델 객체의 state_dict 저장
                max_recall = test_recall
            if test_pre > max_precision:
                torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_precision.pt'))   # 모델 객체의 state_dict 저장
                max_precision = test_pre
            if test_f1 > self.max_f1:
                torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_f1_score.pt'))   # 모델 객체의 state_dict 저장
                self.max_f1 = test_f1
            
            # self.early_stopping(test_loss, self.model)
            # if self.early_stopping.early_stop:
            #     logging.info(f'__ Early Stopping __')

            # else:
            #     if test_loss < min_loss:
            #         torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_Loss.pt'))   # 모델 객체의 state_dict 저장
            #         min_loss = test_loss    
            #     if test_recall > max_recall:
            #         torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_recall.pt'))   # 모델 객체의 state_dict 저장
            #         max_recall = test_recall
            #     if test_pre > max_precision:
            #         torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_precision.pt'))   # 모델 객체의 state_dict 저장
            #         max_precision = test_pre
            #     if test_f1 > max_f1:
            #         torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'Multi-label_best_f1_score.pt'))   # 모델 객체의 state_dict 저장
            #         max_f1 = test_f1
            print()
            
        self.test_frame.to_csv(os.path.join(self.save_model_path, 'test.csv'))
        logging.info(f'\n\n____ Finish Fine-Tuning: [Loss] {min_loss:.3f}\t[Recall] {max_recall:.3f}\t[Precision] {max_precision:.3f}\t[F1_score] {self.max_f1:.3f}____')
        return

    def Average(self, lst):
        return sum(lst) / len(lst)

    def cls_matrix(self, epoch, Predicts, Labels):
        if self.args.num_cls == 2:
            cls_list = {0:'C', 1:'Cut'}
        else:
            cls_list = {0:'A', 1:'HSM', 2:'C', 3:'Cut', 4:'Wire'}

        acc         = []
        precision   = []
        recall      = []
        f1          = []

        for i in range(self.args.num_cls):
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
        
        acc = self.Average(acc)
        precision = self.Average(precision)
        recall = self.Average(recall)
        f1 = self.Average(f1)
        print()
        print(f'{epoch} Epoch Test Accuracy:{acc:.3f}\t Precision:{precision:.3f}\t Recall:{recall:.3f}\t F1_score:{f1:.3f}\t Max_F1_score:{self.max_f1:.3f}')
        print()

        data = [round(i, 3) for i in [epoch, acc, precision, recall, f1]]
        data = pd.DataFrame(data=[data], columns=['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1_score'])
        self.test_frame = pd.concat([self.test_frame, data], axis=0)
        return recall, precision, f1

    @torch.no_grad()
    def test(self, epoch):
        self.model.eval()

        Predicts = torch.rand(size=(0, self.args.num_cls)).to(self.device)
        Labels   = torch.rand(size=(0, self.args.num_cls)).to(self.device)
        test_loss = 0
        iter = 0
        for iter, (images, targets, name) in enumerate(self.test_data):
            images  = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss    = self.criterion(outputs, targets)
            test_loss += loss.item()

            # self.writer.add_scalar('Test_Loss', loss.item(), global_step=iter)
            outputs = outputs > 0.5            
            Predicts = torch.cat([Predicts, outputs.long()], dim=0)
            Labels   = torch.cat([Labels, targets], dim=0)


        test_recall, test_pre, test_f1 = self.cls_matrix(epoch, Predicts, Labels)
        
        self.writer.add_scalar('Test_loss', loss.item(), global_step=epoch)
        self.writer.add_scalar('Test_recall', test_recall, global_step=epoch)

        return test_loss, test_recall, test_pre, test_f1
