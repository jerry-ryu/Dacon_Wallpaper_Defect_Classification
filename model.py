import torch
import torch.nn as nn
import torch.nn.functional as F

# model size

from torchvision.models import efficientnet_b0
class Efficientnet_b0(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_b0,self).__init__()
        self.backbone = efficientnet_b0(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
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
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(1000,num_classes)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

from torchvision.models import efficientnet_v2_l
class Efficientnet_v2_l(nn.Module):
    def __init__(self, num_classes):
        super(Efficientnet_v2_l,self).__init__()
        self.backbone = efficientnet_v2_l(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

from torchvision.models import swin_b
class Swin_b(nn.Module):
    def __init__(self, num_classes):
        super(Swin_b,self).__init__()
        self.backbone = swin_b(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

from torchvision.models import convnext_large
class Convnext_large(nn.Module):
    def __init__(self, num_classes):
        super(Convnext_large,self).__init__()
        self.backbone = convnext_large(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(1000,num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# layer

from torchvision.models import convnext_large
class Layer_2(nn.Module):
    def __init__(self, num_classes):
        super(Layer_2,self).__init__()
        self.backbone = convnext_large(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,num_classes)
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

from torchvision.models import convnext_large
class Layer_3(nn.Module):
    def __init__(self, num_classes):
        super(Layer_3,self).__init__()
        self.backbone = convnext_large(weights='DEFAULT')
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,1000),
            nn.ReLU(),
            nn.Linear(1000,num_classes)
            )
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

# list
_model_entrypoints = {
    #model size
    'efficientnet_b0': Efficientnet_b0,
    'swin_t': Swin_t,
    'efficientnet_v2_l': Efficientnet_v2_l,
    'swin_b': Swin_b,
    'convnext_large': Convnext_large,
    
    #layer
    'layer_2': Layer_2,
    'layer_3': Layer_3

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