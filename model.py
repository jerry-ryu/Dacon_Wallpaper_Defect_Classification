import torch
import torch.nn as nn
import torch.nn.functional as F

# model

from torchvision.models import efficientnet_b0
class Efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b0,self).__init__()
        self.backbone = efficientnet_b0(weights='DEFAULT')
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

from torchvision.models import swin_t
class Swin_t(nn.Module):
    def __init__(self, num_classes):
        super(Swin_t,self).__init__()
        self.backbone = swin_t(weights='DEFAULT')
        self.classifier = nn.Linear(1000,num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# list
_model_entrypoints = {
    'efficientnet_b0': Efficientnet_b0,
    'swin_t': Swin_t

}

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]


def is_model(model_name):
    return model_name in _model_entrypoints


def create_model(model_name, **kwargs):
    if is_model(model_name):
        create_fn = model_entrypoint(model_name)
        model = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % model_name)
    return model