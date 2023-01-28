import logging
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils import save_config_file
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BYOL_Train(object):
    def __init__(self, *args, **kwargs):
        self.epoch      = kwargs['epoch']
        self.args       = kwargs['args']
        self.dataset    = kwargs['data']
        self.save_path  = kwargs['save']
        self.device     = kwargs['device']
        self.learner    = kwargs['learner']
        self.model      = kwargs['backbone']
        self.writer     = SummaryWriter(self.save_path)
        self.optimizer  = torch.optim.Adam(self.learner.parameters(), lr=3e-4)        

        logging.basicConfig(filename=os.path.join(self.save_path, 'training.log'), level=logging.DEBUG)
        logging.getLogger('matplotlib.font_manager').disabled = True      

    def train(self):
        logging.info(f"Start BYOL training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.\n")
        save_config_file(self.save_path, self.args)

        n_iter = 0
        min_loss = 100000
        min_loss_epoch = 100000
        min_epoch = 0
        for epoch in range(self.epoch):
            print(f'__Epoch {epoch} Train Start__')
            epoch_loss = 0
            for images in tqdm(self.dataset):  
                images = images.float().to(self.device)
                loss = self.learner(images)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.learner.update_moving_average()

                str = f'\t[Epoch:{epoch}][{n_iter}||{len(self.dataset)}]\t BYOL loss: {loss:.5f}'
                logging.info(str)
                self.writer.add_scalar('BYOL_loss', loss, global_step=n_iter)

                if loss.item() < min_loss:
                    min_loss = loss
                    torch.save(self.model.state_dict(), os.path.join(self.save_path, 'BYOL_improved-net.pt'))

                epoch_loss += loss.item()
                n_iter += 1
                if n_iter % 5 == 0:
                    print(str)
            
            print(f'{epoch} Epoch__BYOL Loss: {epoch_loss:.3f}\n')
            if epoch_loss < min_loss_epoch:
                min_epoch = epoch
                min_loss_epoch = min_epoch