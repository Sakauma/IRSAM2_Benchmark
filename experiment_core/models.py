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
    def __init__(self):
        super().__init__()
        final_conv = nn.Conv2d(16, 1, kernel_size=1)
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
        features = torch.cat([teacher_logits, box_prior], dim=1)
        encoded = self.encoder(features)
        residual = self.decoder(encoded)
        if residual.shape[-2:] != teacher_logits.shape[-2:]:
            residual = F.interpolate(residual, size=teacher_logits.shape[-2:], mode="bilinear", align_corners=False)
        return teacher_logits + residual


class TinyPIDNetS(nn.Module):
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
        return F.interpolate(self.classifier(feat), size=x.shape[-2:], mode="bilinear", align_corners=False)


class TinyEncoderDecoder(nn.Module):
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
    def __init__(self):
        super().__init__()
        if SegformerConfig is None or SegformerForSemanticSegmentation is None:
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
