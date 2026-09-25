"""Сеть из §3. После каждой свёртки ReLU."""
import torch
from torch import nn


def build_model(num_classes: int = 100) -> nn.Sequential:
    def conv(c_in, c_out, k, stride=1):
        return nn.Conv2d(c_in, c_out, kernel_size=k, stride=stride, padding=k // 2, bias=False)

    return nn.Sequential(
        conv(3, 32, 7, 2), nn.ReLU(inplace=True),          # S/2
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),   # S/4
        conv(32, 64, 5), nn.ReLU(inplace=True),             # S/4
        conv(64, 128, 3, 2), nn.ReLU(inplace=True),         # S/8
        conv(128, 256, 1), nn.ReLU(inplace=True),           # S/8
        conv(256, 256, 3, 2), nn.ReLU(inplace=True),        # S/16
        conv(256, 512, 1), nn.ReLU(inplace=True),           # S/16
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),              # B x 512
        nn.Linear(512, 256), nn.ReLU(inplace=True),
        nn.Linear(256, num_classes),
    )
