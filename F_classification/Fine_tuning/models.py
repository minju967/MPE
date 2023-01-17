import torch
import torch.nn as nn

from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, resnet18, resnet34

class finetuning_Model(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        if model_name == 'Resnet18':
            self.model = resnet18(pretrained=True)
            self.model.fc = nn.Linear(512, 6)
        elif model_name == 'Resnet34':
            self.model = resnet34(pretrained=True)
            self.model.fc = nn.Linear(512, 6)
        elif model_name == 'Resnet50':
            self.model = resnet50(pretrained=True)
            self.classifier = nn.Linear(1000, 6)
        elif model_name == 'EfficientNet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
            self.classifier = nn.Linear(1000, 6)

    def forward(self, x):
        if self.model_name == 'Resnet50':
            x = self.model(x)
            y = self.classifier(x)
        elif self.model_name == 'Resnet18':
            y = self.model(x)
        elif self.model_name == 'Resnet34':
            y = self.model(x)
        elif self.model_name == 'EfficientNet':
            x = F.relu(self.model(x))
            y = self.classifier(x)
        return y

