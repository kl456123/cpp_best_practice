# -*- coding: utf-8 -*-

import os
import torch
from torchvision.models import resnet


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.batchnorm = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = torch.nn.Flatten(1)
        # self.fc = linear();

    def forward(self, x):
        # x = self.conv2d1(x)
        # x = self.batchnorm(x)
        #  x = self.relu(x)
        # x = self.maxpool(x)
        #  x = self.maxpool(x) + x
        x = self.flatten(x)
        return x


def generate_onnx(saved_path):
    """
    Generate a pth model to used for test
    Args:
        saved_path: string, specify where to store the model
    """
    # build graph first
    inputs = torch.ones(1, 3, 224, 224)

    # model construction
    #  model = Model()
    model = resnet.resnet50()
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
    print(res)

    # save in any time
    torch.save(model.state_dict(), pth_path)

    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        inputs,
        saved_path,
        verbose=True,
        output_names=output_names,
        input_names=input_names)
    # print(model(inputs).shape)


if __name__ == '__main__':
    generate_onnx('./demo.onnx')
