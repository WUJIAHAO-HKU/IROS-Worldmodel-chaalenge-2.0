"""Minimal LoRA injection for the official IRASim transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        self.scale = float(alpha) / rank

    def forward(self, value):
        return self.base(value) + self.lora_b(self.lora_a(value)) * self.scale


def inject_irasim_lora(model, rank: int = 8, alpha: float = 8.0) -> list[str]:
    names = []
    for index, block in enumerate(model.blocks):
        for parent_name, parent, child_name in (
            ("attn", block.attn, "qkv"), ("attn", block.attn, "proj"),
            ("mlp", block.mlp, "fc1"), ("mlp", block.mlp, "fc2"),
            ("adaLN_modulation", block.adaLN_modulation, str(len(block.adaLN_modulation) - 1)),
        ):
            child_index = int(child_name) if child_name.isdigit() else child_name
            original = parent[child_index] if isinstance(child_index, int) else getattr(parent, child_index)
            replacement = LoRALinear(original, rank, alpha)
            if isinstance(child_index, int): parent[child_index] = replacement
            else: setattr(parent, child_index, replacement)
            names.append(f"blocks.{index}.{parent_name}.{child_name}")
    original = model.final_layer.adaLN_modulation[-1]
    model.final_layer.adaLN_modulation[-1] = LoRALinear(original, rank, alpha)
    names.append("final_layer.adaLN_modulation.1")
    return names
