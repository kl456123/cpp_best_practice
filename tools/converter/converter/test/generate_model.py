# -*- coding: utf-8 -*-

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d1 = torch.nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        # self.conv2d2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv2d1(x)
        # x = self.conv2d2(x)
        return x


def generate_onnx(saved_path):
    """
    Generate a pth model to used for test
    Args:
        saved_path: string, specify where to store the model
    """
    # build graph first
    inputs = torch.ones(1, 3, 3, 3)
    model = Model()

    input_names = ['input']
    output_names = ['output']
    torch.onnx.export(
        model,
        inputs,
        saved_path,
        verbose=True,
        output_names=output_names,
        input_names=input_names)
    print(model(inputs))


if __name__ == '__main__':
    generate_onnx('./demo.onnx')
