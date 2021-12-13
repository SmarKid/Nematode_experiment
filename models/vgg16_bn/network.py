import torchvision
import torch
from torchvision.models.vgg import VGG, vgg16_bn, make_layers, cfgs, load_state_dict_from_url
from torch.nn import init


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

def network(pretrained_path=None, batch_norm=True):
    model = VGG(make_layers(cfgs['D'], batch_norm=batch_norm), num_classes=30)
    if pretrained_path:
        state_dict = torch.load(pretrained_path)
        del state_dict['classifier.6.bias']
        del state_dict['classifier.6.weight']
        model.load_state_dict(state_dict, strict=False)
        init.normal_(model.classifier[6].weight, mean=0, std=0.01)
        init.constant_(model.classifier[6].bias, val=0)

    return model
