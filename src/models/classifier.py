"""Backbone wrappers for VGG-16 and InceptionV3, paper-faithful classifiers for Project 3.

Both backbones are loaded with ImageNet pretrained weights (transfer learning). The final
classifier is replaced by a Linear(features, num_classes). Loss is multi-label BCE
(applied externally), so the head emits raw logits.

InceptionV3 is built with aux_logits=False to keep the forward signature simple
(single tensor output). aux_logits typically gives a small boost during training but
adds plumbing; the paper does not mention using it.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    Inception_V3_Weights,
    VGG16_Weights,
    densenet121,
    inception_v3,
    vgg16,
)

SUPPORTED = ("densenet121", "vgg16", "inception_v3")


def build_classifier(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "densenet121":
        # DenseNet-121: CheXNet-style medical-imaging standard. Dense "feature reuse" preserves
        # faint low-level cues (osteophytes, disc-space narrowing) across depth — well suited to
        # 128px grayscale ROI patches. Replace the final classifier with a num_classes head.
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        return model
    if name == "vgg16":
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = vgg16(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return model
    if name == "inception_v3":
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        model = inception_v3(weights=weights)  
        model.aux_logits = False             
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"unknown backbone {name!r}; supported: {SUPPORTED}")
