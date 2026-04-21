from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

SUPPORTED_ARCHES = ("seresnet34", "resnet34")


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=True),
            nn.Sigmoid(),
        )
        self._init_identity_bias()

    def _init_identity_bias(self) -> None:
        # 让初始门控接近恒等映射（2*sigmoid(0)=1），降低训练初期的特征抑制。
        first_linear = self.fc[0]
        second_linear = self.fc[2]
        first_linear._skip_default_init = True  # type: ignore[attr-defined]
        second_linear._skip_default_init = True  # type: ignore[attr-defined]
        nn.init.kaiming_uniform_(first_linear.weight, a=1.0)
        nn.init.zeros_(first_linear.bias)
        nn.init.zeros_(second_linear.weight)
        nn.init.zeros_(second_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        weights = self.pool(x).view(b, c)
        weights = self.fc(weights).view(b, c, 1, 1)
        return x * (2.0 * weights)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, reduction: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out = out + identity
        out = self.relu(out)
        return out


class _BaseResNet34(nn.Module):
    block_cls: type[nn.Module]

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.in_channels = 64

        # ResNet-D 风格 stem：3x3 堆叠，通常比单个 7x7 对中小数据集更友好。
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(64, blocks=3, stride=1)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.layer3 = self._make_layer(256, blocks=6, stride=2)
        self.layer4 = self._make_layer(512, blocks=3, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.neck = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
        )
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        raise NotImplementedError

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [self._make_block(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(self._make_block(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                if getattr(module, "_skip_default_init", False):
                    continue
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.neck(x)
        return self.fc(x)


class SEResNet34(_BaseResNet34):
    """手写版 ResNet34 + SE（不依赖 torchvision.models）。"""

    def __init__(self, num_classes: int, reduction: int = 16) -> None:
        self.reduction = reduction
        super().__init__(num_classes=num_classes)

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        return SEBasicBlock(in_channels, out_channels, stride=stride, reduction=self.reduction)


class ResNet34(_BaseResNet34):
    """手写版标准 ResNet34（无注意力机制）。"""

    def _make_block(self, in_channels: int, out_channels: int, stride: int) -> nn.Module:
        return BasicBlock(in_channels, out_channels, stride=stride)


def _load_torchvision_imagenet_weights(model: nn.Module) -> None:
    weights = ResNet34_Weights.IMAGENET1K_V1
    tv_state_dict = resnet34(weights=weights).state_dict()

    remapped: dict[str, torch.Tensor] = {}
    for key, value in tv_state_dict.items():
        if key.startswith("fc."):
            continue
        if key == "conv1.weight":
            remapped["stem.0.weight"] = value
            continue
        if key.startswith("bn1."):
            remapped[key.replace("bn1.", "stem.1.", 1)] = value
            continue
        remapped[key] = value

    model.load_state_dict(remapped, strict=False)


def build_model(num_classes: int, arch: str = "seresnet34", pretrained: bool = False) -> nn.Module:
    arch = arch.lower()
    model: nn.Module
    if arch == "seresnet34":
        model = SEResNet34(num_classes=num_classes)
    elif arch == "resnet34":
        model = ResNet34(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型架构: {arch}，可选: {', '.join(SUPPORTED_ARCHES)}")

    if pretrained:
        _load_torchvision_imagenet_weights(model)
    return model
