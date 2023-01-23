import os
import sys
import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18
from trainer import BYOLTrainer

sys.path.append(os.path.dirname('F_classification'))
from F_classification.create_dataset import PyTorch_BYOL


print(torch.__version__)
torch.manual_seed(0)
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def main():
    conf_path ='D:\\VS_CODE\\Paper\\F_classification\\PyTorch-BYOL\\config\\config.yaml'
    config = yaml.load(open(conf_path, "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])
    path = 'D:\\VS_CODE\\Paper\\Datafolder\\dataset_0'
    train_dataset = PyTorch_BYOL(os.path.join(path), phase='Train', transform=data_transform)
    
    # online network
    online_network = ResNet18(**config['network']).to(device)

    # predictor network
    predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    target_network = ResNet18(**config['network']).to(device)

    optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])

    trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    trainer.train(train_dataset)


if __name__ == '__main__':
    main()
