"""轻量模型组件定义。

当前 benchmark 里，真正重模型的部分是外部 SAM2 teacher。
这里实现的是围绕 benchmark 需要的轻量 student / adapter 结构。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except Exception:  # pragma: no cover - optional dependency
    SegformerConfig = None
    SegformerForSemanticSegmentation = None


class ConvBlock(nn.Module):
    """两层卷积的小块，作为轻量 encoder/decoder 的基础单元。"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PromptConditionedMaskAdapter(nn.Module):
    """对 SAM2 logits 做残差修正的 adapter。

    输入由两部分组成：
    1. teacher logits
    2. box prior

    输出是对 teacher logits 的残差修正，而不是从零开始预测 mask。
    """

    def __init__(self):
        super().__init__()
        final_conv = nn.Conv2d(16, 1, kernel_size=1)
        # 最后一层零初始化，让训练初期模型先退化为“基本不改 teacher 输出”。
        nn.init.zeros_(final_conv.weight)
        nn.init.zeros_(final_conv.bias)
        self.encoder = nn.Sequential(
            ConvBlock(2, 16),
            nn.MaxPool2d(2),
            ConvBlock(16, 32),
            nn.MaxPool2d(2),
            ConvBlock(32, 64),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            ConvBlock(16, 16),
            final_conv,
        )

    def forward(self, teacher_logits: torch.Tensor, box_prior: torch.Tensor) -> torch.Tensor:
        # 把 prompt 信息显式拼接进网络，避免 adapter 只看到 teacher 输出而忽略几何条件。
        features = torch.cat([teacher_logits, box_prior], dim=1)
        encoded = self.encoder(features)
        residual = self.decoder(encoded)
        if residual.shape[-2:] != teacher_logits.shape[-2:]:
            residual = F.interpolate(residual, size=teacher_logits.shape[-2:], mode="bilinear", align_corners=False)
        return teacher_logits + residual


class TinyPIDNetS(nn.Module):
    """PIDNet-S 风格的极简替代实现，用作 control baseline。"""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128),
        )
        self.classifier = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(self.stem(x))
        # 输出统一上采样回输入分辨率，方便和 GT mask 对齐。
        return F.interpolate(self.classifier(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)


class TinyEncoderDecoder(nn.Module):
    """当 transformers 不可用时的 SegFormer 退化替代。"""

    def __init__(self):
        super().__init__()
        self.down1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.mid = ConvBlock(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = self.down1(x)
        skip2 = self.down2(self.pool1(skip1))
        mid = self.mid(self.pool2(skip2))
        up2 = self.up2(mid)
        if up2.shape[-2:] != skip2.shape[-2:]:
            up2 = F.interpolate(up2, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, skip2], dim=1))
        up1 = self.up1(dec2)
        if up1.shape[-2:] != skip1.shape[-2:]:
            up1 = F.interpolate(up1, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, skip1], dim=1))
        return self.head(dec1)


class SegFormerWrapper(nn.Module):
    """SegFormer 包装层。

    如果环境缺少 transformers，则自动回退到轻量 encoder-decoder，
    保证 benchmark 平台在依赖不完整时也能跑 control baseline。
    """

    def __init__(self):
        super().__init__()
        if SegformerConfig is None or SegformerForSemanticSegmentation is None:
            # 依赖缺失时仍保留一个可训练替代，避免方法注册器失效。
            self.backbone = None
            self.fallback = TinyEncoderDecoder()
        else:
            config = SegformerConfig(
                num_labels=1,
                num_channels=3,
                hidden_sizes=[32, 64, 160, 256],
                depths=[2, 2, 2, 2],
                decoder_hidden_size=128,
                classifier_dropout_prob=0.1,
            )
            self.backbone = SegformerForSemanticSegmentation(config)
            self.backbone.config.num_labels = 1
            self.fallback = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is None:
            return self.fallback(x)
        out = self.backbone(pixel_values=x)
        return F.interpolate(out.logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
