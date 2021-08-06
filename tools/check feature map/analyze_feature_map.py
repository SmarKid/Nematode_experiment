import torch
import torchvision
from alexnet_model import AlexNet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
data_transform = transforms.Compose([
    transforms.Resize((400, 600)),
    transforms.ToTensor(),
    normalize
])

# data_transform = transforms.Compose(
#     [transforms.Resize(256),
#      transforms.CenterCrop(224),
#      transforms.ToTensor(),
#      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# create model
model = AlexNet(num_classes=30)
# model = resnet34(num_classes=5)
# load model weights
model_weight_path = "./models/alexnet/weights/epoch_20.pth"  # "./resNet34.pth"
model.load_state_dict(torch.load(model_weight_path)['state_dict'])
print(model)

# load image
img = Image.open("E:\workspace\线虫数据集\分类数据集\图片整理5\\6_0.417_0.500\\b_16_1_10_2_10.JPG")
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# forward
out_put = model(img)
for feature_map in out_put:
    # [N, C, H, W] -> [C, H, W]
    im = np.squeeze(feature_map.detach().numpy())
    # [C, H, W] -> [H, W, C]
    im = np.transpose(im, [1, 2, 0])

    # show top 12 feature maps
    plt.figure()
    for i in range(12):
        ax = plt.subplot(3, 4, i+1)
        # [H, W, C]
        plt.imshow(im[:, :, i], cmap='gray')
    plt.show()
    pass

