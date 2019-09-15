import os
import torch
from torchvision.models import resnet18
from torchsummary import summary
from thop import profile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model = resnet18(pretrained=True)
model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.cuda()
# summary(model, (3, 224, 224))
image_size = (1, 3, 224, 224)
flops, params = profile(model, input_size=image_size)
print(flops)
print(params)
print(image_size)