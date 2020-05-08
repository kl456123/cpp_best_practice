# -*- coding: utf-8 -*-

import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(4, 1, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv2d(x)


def generate_onnx(saved_path):
    """
    Generate a pth model to used for test
    Args:
        saved_path: string, specify where to store the model
    """
    # build graph first
    inputs = torch.ones(1, 4, 224, 224)
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


if __name__ == '__main__':
    generate_onnx('./demo.onnx')
