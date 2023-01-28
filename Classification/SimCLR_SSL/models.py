import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from exceptions.exceptions import InvalidBackboneError
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim, pretrain):
        super(ResNetSimCLR, self).__init__()
        
        if pretrain:
            # pretrain == True
            self.resnet_dict = {"Resnet18": models.resnet18(weights=ResNet18_Weights.DEFAULT),
                                "Resnet34": models.resnet34(weights=ResNet34_Weights.DEFAULT),
                                "Resnet50": models.resnet50(weights=ResNet50_Weights.DEFAULT),
                                "EfficientNetB0":EfficientNet.from_pretrained('efficientnet-b0'),
                                "EfficientNetB1":EfficientNet.from_pretrained('efficientnet-b1')}
        else:
            # pretrain == False
            self.resnet_dict = {"Resnet18": models.resnet18(weights=None, num_classes=out_dim),
                                "Resnet34": models.resnet34(weights=None, num_classes=out_dim),
                                "Resnet50": models.resnet50(weights=None, num_classes=out_dim),
                                "EfficientNetB0":EfficientNet.from_name('efficientnet-b0'),
                                "EfficientNetB1":EfficientNet.from_name('efficientnet-b1')}

        self.backbone = self._get_basemodel(base_model)

        if pretrain:
            if 'Resnet' in base_model:
                dim_mlp = self.backbone.fc.in_features
                self.backbone.fc =  nn.Linear(dim_mlp, out_dim)
            else:
                dim_mlp = self.backbone._fc.in_features
                self.backbone._fc =  nn.Linear(dim_mlp, out_dim)

        # Proejector
        if 'Resnet' in base_model:
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            self.backbone._fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone._fc)
        
    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ResNetBYOL(nn.Module):
    def __init__(self, base_model):
        super(ResNetBYOL, self).__init__()

        # pretrain == True
        self.resnet_dict = {"Resnet18": models.resnet18(weights=ResNet18_Weights.DEFAULT),
                            "Resnet34": models.resnet34(weights=ResNet34_Weights.DEFAULT),
                            "Resnet50": models.resnet50(weights=ResNet50_Weights.DEFAULT)}

        self.backbone = self._get_basemodel(base_model)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model