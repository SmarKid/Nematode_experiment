from torch.hub import load_state_dict_from_url
import torchvision
from torch.nn import init
import torch
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-7be5be79.pth',
}

class Network(torchvision.models.AlexNet):
    def __init__(self) -> None:
        super().__init__(num_classes=30)

def network(pretrained=False):
    model = Network()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'])
        del state_dict['classifier.6.weight']
        del state_dict['classifier.6.bias']
        model.load_state_dict(state_dict, strict=False)
        init.normal_(model.classifier[6].weight, mean=0, std=0.01)
        init.constant_(model.classifier[6].bias, val=0)
    return model


