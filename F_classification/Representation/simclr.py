import logging
import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint

torch.manual_seed(0)
logger = logging.getLogger(__name__)

class SimCLR(object):
    def __init__(self, *args, **kwargs):
        self.device     = kwargs['device']
        self.args       = kwargs['args']
        self.model      = kwargs['model'].to(self.device)
        self.optimizer  = kwargs['optimizer']
        self.scheduler  = kwargs['scheduler']
        self.save_path  = kwargs['save']
        self.writer     = SummaryWriter(self.save_path)
        self.data_frame = pd.DataFrame(index=range(0,0), columns=['Epoch', 'Iteration', 'iter_Loss', 'Top1_acc', 'Top5_acc', 'Epoch_Loss', 'Epoch_acc'])

        logging.basicConfig(filename=os.path.join(self.save_path, 'training.log'), level=logging.DEBUG)
        logging.getLogger('matplotlib.font_manager').disabled = True

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask    = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels  = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature
        return logits, labels

    def show_plot(self, Acc, Loss):
        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(list(range(0,self.args.epochs)), Loss)
        plt.subplot(1,2,2)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(list(range(0,self.args.epochs)), Acc)
        # plt.show()

        plt.savefig(f'{self.save_path}\\figure.png')

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.save_path, self.args)

        n_iter = 0
        max_acc = 0
        max_epoch = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        Train_loss_list = []
        Train_acc_list  = []
        Test_loss_list  = []
        Test_acc_list   = []

        for epoch_counter in range(self.args.epochs):
            print(f'__Epoch {epoch_counter} Train Start__')
            acc  = 0
            train_loss = 0
            for images in tqdm(train_loader):  
                images = torch.cat([images[0]['image'], images[1]['image']], dim=0)
                images = images.float().to(self.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                train_loss += loss.item()
                acc += top1[0].item()
                logging.info(f'loss : {loss:.5f}')
                logging.info(f'acc/top1 : {top1[0]}')
                logging.info(f'acc/top2 : {top5[0]}')

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)

                data = [epoch_counter, n_iter, f'{loss.item():.5f}', f'{top1[0]:.5f}', f'{top5[0]:.5f}', 0, 0]
                df = pd.DataFrame(data=[data], columns=['Epoch', 'Iteration', 'iter_Loss', 'Top1_acc', 'Top5_acc', 'Epoch_Loss', 'Epoch_acc'])
                self.data_frame = pd.concat([self.data_frame, df], ignore_index=True)

                n_iter += 1

            acc = acc/len(train_loader)
            loss = train_loss/len(train_loader)

            data = [epoch_counter, n_iter, 0, 0, 0, f'{loss:.5f}', f'{acc:.5f}']
            df = pd.DataFrame(data=[data], columns=['Epoch', 'Iteration', 'iter_Loss', 'Top1_acc', 'Top5_acc', 'Epoch_Loss', 'Epoch_acc'])
            self.data_frame = pd.concat([self.data_frame, df], ignore_index=True)

            Train_acc_list.append(acc)
            Train_loss_list.append(loss)

            if acc > max_acc:
                max_acc = acc
                max_epoch = epoch_counter
                checkpoint_name = 'checkpoint_best.pth.tar'
                save_checkpoint({
                    'epoch': self.args.epochs,
                    'arch': self.args.backbone,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.save_path, checkpoint_name))
                logging.info(f"Model checkpoint and metadata has been saved at {self.save_path}.")

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}\t Epoch accuracy:{acc:.5f}")

        self.data_frame.to_csv(os.path.join(self.save_path, 'experiment.csv'))

        self.show_plot(Train_acc_list, Train_loss_list)

        logging.info("Training has finished.")
        logging.info(f"Best accuracy Train Epoch is {max_epoch}")
        logging.info(f"Model checkpoint and metadata has been saved at {self.save_path}.")