# -*- coding: utf-8 -*-

import os
import torch
from torchvision.models import resnet


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 10
        out_channels = 5
        self.conv2d1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten(1)
        self.model = resnet.resnet50()
        # self.model.eval()
        self.fc = torch.nn.Linear(10, 5);

    def forward(self, x):
        # x = self.model.conv1(x)
        # x = self.model.bn1(x)
        # x = self.model.relu(x)
        # x = self.model.maxpool(x)
        # x = self.model.layer1(x)
        # x = self.model.layer2(x)
        # x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # x = self.model.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.model.fc(x)
        x = self.conv2d1(x)
        # x = self.batchnorm(x)
        # x = self.avgpool(x)
        # x = self.relu(x)
        # x = self.fc(x)
        # x = self.maxpool(x)
        # x = self.maxpool(x) + x
        # x = self.flatten(x)
        return x


def generate_onnx(saved_path):
    """
    Generate a pth model to used for test
    Args:
        saved_path: string, specify where to store the model
    """
    # build graph first
    size = 3
    inputs = torch.ones(1, 10, 3, 3)

    # model construction
    model = Model()
    # model = resnet.resnet50()
    # inferece works
    model.eval()
    pth_path = '{}.pth'.format(os.path.splitext(saved_path)[0])

    # load weights
    if os.path.exists(pth_path):
        print("load weights from {}".format(pth_path))
        try:
            model.load_state_dict(torch.load(pth_path))
        except RuntimeError:
            print('load weights failed. the model is not initialized')

    # inference
    res = model(inputs)
    # convert from nchw to nhwc
    if len(res.shape) == 4:
        print(res.permute(0, 2, 3, 1))
    else:
        print(res)
    print(res.shape)

    # save in any time
    torch.save(model.state_dict(), pth_path)

    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        inputs,
        saved_path,
        verbose=False,
        output_names=output_names,
        input_names=input_names)
    # print(model(inputs).shape)


if __name__ == '__main__':
    generate_onnx('./demo.onnx')
