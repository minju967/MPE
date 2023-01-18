from time import time

import os
import copy
import torch
import torch.nn          as nn
import logging
import argparse
import pandas            as pd
import torch.optim       as optim
import matplotlib.pyplot as plt


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
        self.test_df     = pd.DataFrame(index=range(0,0), columns=['Idx', 'Name', 'A', 'HSM', 'C', 'Cut', 'Wire', 'None', 'Path'])
        self.train_frame  = pd.DataFrame(index=range(0,0), columns=['Epoch', 'Accuracy'])
        self.test_frame   = pd.DataFrame(index=range(0,0), columns=['Epoch', 'Accuracy'])

    def train(self, epochs):
        max_acc = 0

        for epoch in range(epochs):
            Start_time = time()
            print('='*30)
            print(f'Epoch:{epoch}__Start')

            self.model.train()
            train_acc = 0
            train_loss = 0
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
                train_acc += acc.item()
                train_loss += loss.item()

                logging.info(f'[Train][{i}|{len(self.train_data)}] Loss:{loss.item():.5f} Acc:{acc.item():.5f}')

                if (i+1) % 5 == 0:
                    print(f'[{i}/{len(self.train_data)}]: Loss:{loss.item():.5f}, Acc:{acc.item():.5f}')
                
            End_time = time()
            Learning_time = End_time - Start_time
            m = Learning_time // 60
            s = Learning_time % 60
            print(f'\nLearning Time: {int(m)}min. {int(s)}sec.')

            train_acc = train_acc/len(self.train_data)
            train_loss = train_loss/len(self.train_data)

            data = [epoch, f'{train_acc:.5f}']
            df = pd.DataFrame(data=[data], columns=['Epoch', 'Accuracy'])
            self.train_frame = pd.concat([self.train_frame, df], ignore_index=True)

            test_acc = self.test(epoch=epoch, args=self.args)

            if test_acc > max_acc:
                torch.save(copy.deepcopy(self.model.state_dict()), os.path.join(self.save_model_path, 'model_state_dict.pt'))   # 모델 객체의 state_dict 저장
                self.test_df.to_csv(os.path.join(self.save_model_path, f'result_{epoch}.csv'))
                max_acc = test_acc
            
            self.train_frame.to_csv(os.path.join(self.save_model_path, 'train.csv'))
            self.test_frame.to_csv(os.path.join(self.save_model_path, 'test.csv'))
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
        
        data = [epoch, f'{epoch_test_acc:.5f}']
        df = pd.DataFrame(data=[data], columns=['Epoch', 'Accuracy'])
        self.test_frame = pd.concat([self.test_frame, df], ignore_index=True)

        return epoch_test_acc