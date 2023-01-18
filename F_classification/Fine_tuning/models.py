import torch
import torch.nn as nn

from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models import resnet50, resnet18, resnet34

class finetuning_Model(nn.Module):
    def __init__(self, model_name, model_PT) -> None:
        super().__init__()
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        if model_name == 'Resnet18':
            self.model = resnet18(pretrained=True)
        elif model_name == 'Resnet34':
            self.model = resnet34(pretrained=True)
        elif model_name == 'Resnet50':
            self.model = resnet50(pretrained=True)
        elif model_name == 'EfficientNet':
            self.model = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
        
        if model_name == 'Resnet18' or model_name == 'Resnet34':
            self.model.fc = nn.Linear(512, 6)
        else:
            self.classifier = nn.Linear(1000, 6)
        
        # for name, param in self.model.named_parameters():
        #     if name not in ['fc.weight', 'fc.bias']:
        #         param.requires_grad = False

        if model_PT:
            checkpoint = torch.load(model_PT, map_location=device)
            state_dict = checkpoint['state_dict']
            
            for k in list(state_dict.keys()):
                if k.startswith('backbone.'):
                    if k.startswith('backbone') and not k.startswith('backbone.fc') or not k.startswith('backbone._fc'):
                        state_dict[k[len("backbone."):]] = state_dict[k]
                del state_dict[k]

            log = self.model.load_state_dict(state_dict, strict=False)
            if model_name != 'EfficientNet':
                assert log.missing_keys == ['fc.weight', 'fc.bias']
            else:
                assert log.missing_keys == ['_fc.weight', '_fc.bias']

            # freeze all layers but the last fc
            # for name, param in self.model.named_parameters():
            #     if name not in ['fc.weight', 'fc.bias']:
            #         param.requires_grad = False

            # parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
            # assert len(parameters) == 2  # fc.weight, fc.bias

    def forward(self, x):
        if self.model_name == 'Resnet18' or self.model_name == 'Resnet34':
            y = self.model(x)
        elif self.model_name == 'Resnet50':
            x = self.model(x)
            y = self.classifier(x)
        elif self.model_name == 'EfficientNet':
            x = F.relu(self.model(x))
            y = self.classifier(x)
        return y

