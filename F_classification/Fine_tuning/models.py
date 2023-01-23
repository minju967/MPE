import torch
import torch.nn as nn

from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class finetuning_Model(nn.Module):
    def __init__(self, model_name, model_PT, cls_num) -> None:
        super().__init__()
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        if model_name == 'Resnet18':
            self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'Resnet34':
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
        elif model_name == 'Resnet50':
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'EfficientNetB0':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)

        if 'Resnet' in model_name:
            dim_mlp = self.model.fc.in_features
        else:
            dim_mlp = self.model._fc.in_features

        if model_name == 'Resnet18' or model_name == 'Resnet34':
            self.model.fc = nn.Linear(512, cls_num)
        elif model_name == 'Resnet50':
            # 1000 --> 512 --> 256 --> num_cls
            self.model.fc = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, cls_num), nn.Sigmoid())
        elif model_name == 'EfficientNetB0':
            # 1280 --> 512 --> 256 --> num_cls
            self.model._fc = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, cls_num), nn.Sigmoid())
  
        if model_PT:
            checkpoint = torch.load(model_PT, map_location=device)
            state_dict = checkpoint['model_state_dict']

            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone.') and 'fc.' not in k:
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            log = self.model.load_state_dict(state_dict, strict=False)
            print(log)

    def forward(self, x):
        if self.model_name == 'Resnet18' or self.model_name == 'Resnet34':
            y = self.model(x)
        elif self.model_name == 'Resnet50':
            y = self.model(x)
        else:
            # EfficientNetB0, EfficientNetB1
            y = self.model(x)
        return y

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class ResNet18(torch.nn.Module):
    def __init__(self, model_name, model_PT, cls_num):
        super(ResNet18, self).__init__()
        if model_name == 'Resnet18':
            resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'Resnet34':
            resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.fine_tuning = True
        self.model_PT    = model_PT

        self.encoder    = torch.nn.Sequential(*list(resnet.children())[:-1])
        checkpoint = torch.load(self.model_PT)
        state_dict = checkpoint['online_network_state_dict']

        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                if k.startswith('encoder.') and 'fc.' not in k:
                    print(k)
                    state_dict[k[len("encoder."):]] = state_dict[k]
            del state_dict[k]        

        log = self.encoder.load_state_dict(state_dict, strict=False)
        print(log)
        self.classifier = nn.Sequential(nn.Linear(resnet.fc.in_features, cls_num))

    def forward(self, x):
        y = self.encoder(x)
        h = y.view(y.shape[0], y.shape[1])
        return self.classifier(h)
