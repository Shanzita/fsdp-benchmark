"""Shared utilities for FSDP benchmark scripts."""
import torch
from torchvision import models

MODEL_MAP = {
    "resnet50": models.resnet50,
    "vit_b_16": models.vit_b_16,
    "vit_l_16": models.vit_l_16,
}
MODEL_CHOICES = list(MODEL_MAP.keys())


def get_model(name):
    """Create a model by name with no pretrained weights."""
    return MODEL_MAP[name](weights=None)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
