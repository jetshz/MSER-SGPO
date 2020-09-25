import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable
import numpy as np
import torchvision.models as models

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        resnet50 =models.resnet50(pretrained = True)

        self.layer0 = nn.Sequential(resnet50.conv1, resnet50.bn1, resnet50.relu, resnet50.maxpool)

        self.layer1 = nn.Sequential(resnet50.layer1)

        self.layer2 = nn.Sequential(resnet50.layer2)

        self.layer3 = nn.Sequential(resnet50.layer3)

        self.layer4 = nn.Sequential(resnet50.layer4)

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.classifier = nn.Conv2d(2048, 20, 1, bias=True)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        # self.backbone =nn.ModuleList([self.layer0, self.layer1, self.layer2, self.layer3, self.layer4])
        # self.new = nn.ModuleList([self.classifier])

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = self.dropout7(x)

        x = F.avg_pool2d(x, kernel_size = (x.size(2), x.size(3)), padding =0)

        x = self.classifier(x)

        x = x.view(x.size(0), -1)

        return x

    def cam_forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.layer4(x))
        x = F.conv2d(x, self.classifier.weight)
        x = F.relu(x)
        return x



    def normalize(self, img):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - mean[0]) / std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - mean[1]) / std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - mean[2]) / std[2]

        return proc_img



