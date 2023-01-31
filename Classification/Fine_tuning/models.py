import torch
import torch.nn as nn

from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class finetuning_Model(nn.Module):
    def __init__(self, model_name, model_PT, cls_num, pre_train) -> None:
        super().__init__()
        self.model_name = model_name

        if pre_train == 'True':
            if model_name == 'Resnet18':
                self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
            elif model_name == 'Resnet34':
                self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
            elif model_name == 'Resnet50':
                self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            elif model_name == 'EfficientNetB0':
                self.model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)    
        else:
            if model_name == 'Resnet18':
                self.model = resnet18(weights=None)                
            elif model_name == 'Resnet34':
                self.model = resnet34(weights=None)
            elif model_name == 'Resnet50':
                self.model = resnet50(weights=None)
            elif model_name == 'EfficientNetB0':
                self.model = EfficientNet.from_name('efficientnet-b0')

        if model_name == 'Resnet18' or model_name == 'Resnet34':
            self.model.fc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(256, cls_num))
        elif model_name == 'Resnet50':
            dim_mlp = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, cls_num))
        elif model_name == 'EfficientNetB0':
            dim_mlp = self.model._fc.in_features
            self.model._fc = nn.Sequential(nn.Linear(dim_mlp, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p=0.5), nn.Linear(512, cls_num))
  
        if model_PT is not None:
            print(f'======== Model PT file Loading: {model_PT} ========')
            state_dict = torch.load(model_PT)
            if cls_num == 5:
                for k in list(state_dict.keys()):
                    if k.startswith('model.'):
                        state_dict[k[len("model."):]] = state_dict[k]
                    del state_dict[k]
            else:
                for k in list(state_dict.keys()):
                    if k.startswith('model.') and not 'fc.' in k:
                        state_dict[k[len("model."):]] = state_dict[k]
                    del state_dict[k]

            log = self.model.load_state_dict(state_dict, strict=False)
            print(log)
            print(f'======== Model PT file Load complete ========')
        
        if cls_num == 2:
            for name, param in self.model.named_parameters():
                if not 'fc.' in name: 
                    param.requires_grad = True

    def forward(self, x):
        if self.model_name == 'Resnet18' or self.model_name == 'Resnet34':
            y = self.model(x)
        elif self.model_name == 'Resnet50':
            y = self.model(x)
        else:
            y = self.model(x)
        return y
