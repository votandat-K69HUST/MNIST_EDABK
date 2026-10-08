"""MLP (Multi-Layer Perceptron) - chi dung cac lop Linear co ban (khong CNN, khong mang phuc tap hon).

Input: anh 28x28 (hoac (B,784)) gia tri 0..1.  Output: 11 logit, tuong ung nhan 10..20 (chi so = nhan - 10).
"""
import torch
from torch import nn

NUM_CLASSES = 11
LABEL_OFFSET = 10

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU, "leaky_relu": nn.LeakyReLU}


class MLP(nn.Module):
    def __init__(self, hidden=(512, 256), dropout=0.2, activation="relu", batchnorm=True,
                 input_dropout=0.0, mean=0.0, std=1.0):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        layers = []
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        d = 28 * 28
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, NUM_CLASSES))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.net((x - self.mean) / self.std)


def build_model(mcfg, mean=0.0, std=1.0):
    return MLP(hidden=tuple(mcfg.get("hidden", [512, 256])), dropout=mcfg.get("dropout", 0.2),
               activation=mcfg.get("activation", "relu"), batchnorm=mcfg.get("batchnorm", True),
               input_dropout=mcfg.get("input_dropout", 0.0), mean=mean, std=std)
