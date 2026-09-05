#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""Misc utilities for checkpoint reload/save (adapted from ClearerVoice-Studio)."""

import os

import torch
import numpy as np

try:
    from pesq import pesq  # noqa: F401  (optional)
except Exception:
    pesq = None

EPS = 1e-6


def load_checkpoint(checkpoint_path, use_cuda):
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint


def reload_for_eval(model, checkpoint_dir, use_cuda):
    """Load the last-best (or last) checkpoint pointed to by the pointer files.

    Same behaviour as the original project: tries 'last_best_checkpoint' first,
    then 'last_checkpoint', then the raw .pt file if given directly.
    """
    print("reloading from: {}".format(checkpoint_dir))
    best_name = os.path.join(checkpoint_dir, "last_best_checkpoint")
    ckpt_name = os.path.join(checkpoint_dir, "last_checkpoint")

    if os.path.isfile(checkpoint_dir):  # a direct checkpoint path was provided
        checkpoint_path = checkpoint_dir
    elif os.path.isfile(best_name):
        name = best_name
        with open(name, "r") as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
    elif os.path.isfile(ckpt_name):
        name = ckpt_name
        with open(name, "r") as f:
            model_name = f.readline().strip()
        checkpoint_path = os.path.join(checkpoint_dir, model_name)
    else:
        print("Warning: There is no exited checkpoint or best_model!!!!!!!!!!!!")
        return
    print("checkpoint_path: {}".format(checkpoint_path))
    checkpoint = load_checkpoint(checkpoint_path, use_cuda)

    if "model" in checkpoint:
        pretrained_model = checkpoint["model"]
    else:
        pretrained_model = checkpoint

    state = model.state_dict()
    for key in state.keys():
        if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
            state[key] = pretrained_model[key]
        elif key.replace("module.", "") in pretrained_model and state[key].shape == pretrained_model[key.replace("module.", "")].shape:
            state[key] = pretrained_model[key.replace("module.", "")]
        elif "module." + key in pretrained_model and state[key].shape == pretrained_model["module." + key].shape:
            state[key] = pretrained_model["module." + key]
        else:
            print("{} not loaded".format(key))
    model.load_state_dict(state)
    print("=> Reload well-trained model for decoding.")


def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, mode="checkpoint"):
    checkpoint_path = os.path.join(
        checkpoint_dir, "model.ckpt-{}-{}.pt".format(epoch, step))
    torch.save(
        {"model": model.state_dict(),
         "optimizer": optimizer.state_dict(),
         "epoch": epoch,
         "step": step},
        checkpoint_path,
    )
    with open(os.path.join(checkpoint_dir, mode), "w") as f:
        f.write("model.ckpt-{}-{}.pt".format(epoch, step))
    print("=> Save checkpoint:", checkpoint_path)


def setup_lr(opt, lr):
    for param_group in opt.param_groups:
        param_group["lr"] = lr
