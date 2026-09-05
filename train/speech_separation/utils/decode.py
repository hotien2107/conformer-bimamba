#!/usr/bin/env python -u
# -*- coding: utf-8 -*-
"""Segmental decoding (adapted from ClearerVoice-Studio utils/decode.py).

Adds the HybridBiMamba_* branch: when the model uses F0 conditioning, the F0
contour of the *input mixture* is extracted once (PyWorld) and sliced to each
decoding segment before being fed to the model.
"""

import os

import torch
import numpy as np

F0_FRAME_PERIOD_S = 0.01  # 10 ms -> 100 fps


def decode_one_audio(model, device, inputs, args):
    if args.network == "MossFormer2_SS_16K":
        return decode_one_audio_mossformer2_ss_16k(model, device, inputs, args)
    elif args.network in ["HybridBiMamba_SS_16K", "HybridBiMamba_SS_8K"]:
        return decode_one_audio_hybrid_bimamba_ss(model, device, inputs, args)
    else:
        print("in decode, {} is not supported!".format(args.network))
        raise ValueError("unknown network {}".format(args.network))


def _extract_f0(wave, sampling_rate):
    """PyWorld F0 of a full utterance; None if pyworld is unavailable."""
    try:
        import pyworld

        f0, t = pyworld.dio(
            wave.astype(np.float64), sampling_rate,
            f0_floor=60.0, f0_ceil=400.0, frame_period=F0_FRAME_PERIOD_S * 1000,
        )
        f0 = pyworld.stonemask(wave.astype(np.float64), f0, t, sampling_rate)
        return f0.astype(np.float32)  # (n_frames,) Hz, 0 = unvoiced
    except Exception as e:
        print("F0 extraction skipped (pyworld unavailable): {}".format(e))
        return None


def _slice_f0(f0_full, st_idx, window_len, sampling_rate):
    """Slice the full-utterance F0 contour to samples [st_idx, st_idx+window)."""
    if f0_full is None:
        return None
    hop = int(sampling_rate * F0_FRAME_PERIOD_S)
    fr0 = st_idx // hop
    fr1 = int(np.ceil((st_idx + window_len) / float(hop)))
    seg = f0_full[fr0:fr1]
    if seg.shape[0] < 1:
        return None
    return seg


def decode_one_audio_mossformer2_ss_16k(model, device, inputs, args):
    """Identical to the original ClearerVoice-Studio MossFormer2_SS decoder."""
    out = []
    decode_do_segement = False
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > window * args.one_time_decode_length:
        print("The sequence is longer than {} seconds, using segmentation decoding...".format(
            args.one_time_decode_length))
        decode_do_segement = True

    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape

    if decode_do_segement:
        outputs = np.zeros((args.num_spks, t))
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx : current_idx + window]
            tmp_out_list = model(tmp_input)
            for spk in range(args.num_spks):
                tmp_out_list[spk] = tmp_out_list[spk][0, :].cpu().numpy()
                if current_idx == 0:
                    outputs[spk, current_idx : current_idx + window - give_up_length] = tmp_out_list[spk][:-give_up_length]
                else:
                    outputs[spk, current_idx + give_up_length : current_idx + window - give_up_length] = tmp_out_list[spk][give_up_length:-give_up_length]
            current_idx += stride
        for spk in range(args.num_spks):
            out.append(outputs[spk, :])
    else:
        out_list = model(inputs)
        for spk in range(args.num_spks):
            out.append(out_list[spk][0, :].cpu().numpy())

    max_abs = 0
    for spk in range(args.num_spks):
        if max_abs < max(abs(out[spk])):
            max_abs = max(abs(out[spk]))
    for spk in range(args.num_spks):
        out[spk] = out[spk] / max_abs
    return out


def decode_one_audio_hybrid_bimamba_ss(model, device, inputs, args):
    """Decoder for HybridBiMamba_SS. Same segmental scheme as MossFormer2_SS,
    additionally slicing the mixture-F0 contour into each decoding window."""
    out = []
    use_f0 = bool(getattr(args, "use_f0", 0))
    decode_do_segement = False
    window = args.sampling_rate * args.decode_window
    stride = int(window * 0.75)
    b, t = inputs.shape
    if t > window * args.one_time_decode_length:
        print("The sequence is longer than {} seconds, using segmentation decoding...".format(
            args.one_time_decode_length))
        decode_do_segement = True

    f0_full = _extract_f0(inputs[0], args.sampling_rate) if use_f0 else None

    if t < window:
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], window - t))], 1)
    elif t < window + stride:
        padding = window + stride - t
        inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    else:
        if (t - window) % stride != 0:
            padding = t - (t - window) // stride * stride
            inputs = np.concatenate([inputs, np.zeros((inputs.shape[0], padding))], 1)
    inputs = torch.from_numpy(np.float32(inputs))
    inputs = inputs.to(device)
    b, t = inputs.shape

    if decode_do_segement:
        outputs = np.zeros((args.num_spks, t))
        give_up_length = (window - stride) // 2
        current_idx = 0
        while current_idx + window <= t:
            tmp_input = inputs[:, current_idx : current_idx + window]
            f0_seg = _slice_f0(f0_full, current_idx, window, args.sampling_rate)
            if f0_seg is not None:
                tmp_out_list = model(tmp_input, torch.from_numpy(f0_seg).float().unsqueeze(0).to(device))
            else:
                tmp_out_list = model(tmp_input)
            for spk in range(args.num_spks):
                tmp_out_list[spk] = tmp_out_list[spk][0, :].cpu().numpy()
                if current_idx == 0:
                    outputs[spk, current_idx : current_idx + window - give_up_length] = tmp_out_list[spk][:-give_up_length]
                else:
                    outputs[spk, current_idx + give_up_length : current_idx + window - give_up_length] = tmp_out_list[spk][give_up_length:-give_up_length]
            current_idx += stride
        for spk in range(args.num_spks):
            out.append(outputs[spk, :])
    else:
        if f0_full is not None:
            out_list = model(inputs, torch.from_numpy(f0_full).float().unsqueeze(0).to(device))
        else:
            out_list = model(inputs)
        for spk in range(args.num_spks):
            out.append(out_list[spk][0, :].cpu().numpy())

    max_abs = 0
    for spk in range(args.num_spks):
        if max_abs < max(abs(out[spk])):
            max_abs = max(abs(out[spk]))
    for spk in range(args.num_spks):
        out[spk] = out[spk] / max_abs
    return out
