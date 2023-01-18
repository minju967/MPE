import torch.nn as nn
import torchvision.models as models

from efficientnet_pytorch import EfficientNet
from exceptions.exceptions import InvalidBackboneError

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"Resnet18": models.resnet18(pretrained=False, num_classes=out_dim),
                            "Resnet34": models.resnet34(pretrained=False, num_classes=out_dim),
                            "Resnet50": models.resnet50(pretrained=False, num_classes=out_dim),
                            "EfficientNet":EfficientNet.from_name('efficientnet-b0')}

        self.backbone = self._get_basemodel(base_model)

        if 'Resnet' in base_model:
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            dim_mlp = self.backbone._fc.in_features
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

