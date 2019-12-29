# -*- coding: utf-8 -*-

import torch
import torch.nn as nn



input = torch.ones(1,3,5,5)
weight = torch.ones(3,3,3,3)
bias = torch.ones(3)
stride = 2
pad = 2
dilation = 2;
output = nn.functional.conv2d(input, weight, bias, stride=stride, padding=pad, dilation=dilation)
print(output)
# print(input.shape)
