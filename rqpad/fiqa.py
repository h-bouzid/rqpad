"""Frozen face image quality assessment (FIQA) network.

Minimal re-implementation of the AI-KD DifFIQA(R) model: an IResNet-100
backbone (InsightFace) followed by a small MLP quality head. Module and
parameter names match the released ``aikd_diffiqar_model.pth`` checkpoint so
it loads with ``strict=True``.

The backbone expects 112x112 RGB faces normalised to [-1, 1] and returns a
512-d embedding; the head maps that embedding to a scalar quality score.
"""
from typing import Optional, Tuple

import torch
from torch import nn


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class IBasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn3(self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x))))))
        return out + identity


class IResNet(nn.Module):
    fc_scale = 7 * 7  # spatial size of the last feature map for 112x112 inputs

    def __init__(self, layers: Tuple[int, int, int, int], num_features: int = 512):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1])
        self.layer3 = self._make_layer(256, layers[2])
        self.layer4 = self._make_layer(512, layers[3])
        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)

    def _make_layer(self, planes: int, blocks: int) -> nn.Sequential:
        downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(planes, eps=1e-05),
        )
        layers = [IBasicBlock(self.inplanes, planes, stride=2, downsample=downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.features(self.fc(torch.flatten(self.bn2(x), 1)))


class QualityHead(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 1024, out_dim: int = 1):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.ac = nn.GELU()
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.ac(self.l1(x)))


class FIQAModel(nn.Module):
    """Backbone + quality head. ``forward`` returns ``(embedding, quality)``."""

    def __init__(self):
        super().__init__()
        self.feature_extractor = IResNet(layers=(3, 13, 30, 3))  # IResNet-100
        self.quality_head = QualityHead()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.feature_extractor(x)
        return embedding, self.quality_head(embedding)


def load_fiqa(weights_path: str) -> FIQAModel:
    """Build the FIQA model and load the pretrained AI-KD DifFIQA(R) weights."""
    model = FIQAModel()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model
