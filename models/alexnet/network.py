import torchvision
class Network(torchvision.models.AlexNet):
    def __init__(self) -> None:
        super().__init__(num_classes=30)
