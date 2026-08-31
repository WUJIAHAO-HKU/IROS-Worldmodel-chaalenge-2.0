#!/usr/bin/env python3
"""Measure one real adapter backward pass for the full IRASim-XL/2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from wam_pipeline.irasim_lora import inject_irasim_lora


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--irasim-root", required=True); parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1); parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--train-temporal-blocks", action="store_true")
    args = parser.parse_args(); sys.path.insert(0, str(Path(args.irasim_root).resolve()))
    from diffusion import create_mask_diffusion
    from models.irasim import IRASim_models
    config = SimpleNamespace(dataset="bridge", state_dim=14, final_frame_ada=False, gradient_checkpointing=not args.no_gradient_checkpointing)
    model = IRASim_models["IRASim-XL/2"](input_size=32, num_frames=13, learn_sigma=False, extras=3, attention_mode="sdpa", args=config)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    for parameter in model.embed_state.parameters(): parameter.requires_grad_(True)
    for parameter in model.mask_emb_fn.parameters(): parameter.requires_grad_(True)
    inject_irasim_lora(model, args.lora_rank, float(args.lora_rank))
    if args.train_temporal_blocks:
        for index in range(1, len(model.blocks), 2):
            for parameter in model.blocks[index].parameters(): parameter.requires_grad_(True)
        for parameter in model.final_layer.parameters(): parameter.requires_grad_(True)
    device=torch.device(args.device); model.to(device).train(); diffusion=create_mask_diffusion("",learn_sigma=False)
    latent=torch.randn(args.batch_size,13,4,32,32,device=device); action=torch.randn(args.batch_size,12,14,device=device); t=torch.randint(0,1000,(args.batch_size,),device=device)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=="cuda"):
        loss=diffusion.training_losses(model,latent,t,{"actions":action,"mask_frame_num":5})["loss"].mean()
    loss.backward()
    print(json.dumps({"loss":float(loss),"parameters":sum(p.numel() for p in model.parameters()),"trainable":sum(p.numel() for p in model.parameters() if p.requires_grad),"allocated_mb":torch.cuda.memory_allocated()/2**20,"peak_mb":torch.cuda.max_memory_allocated()/2**20}))


if __name__ == "__main__": main()
