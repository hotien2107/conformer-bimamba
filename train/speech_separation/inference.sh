#!/bin/bash
# Inference with a trained separation model.
#   network=HybridBiMamba_SS_16K (use_f0: 1 in the config for F0 conditioning)
#   network=MossFormer2_SS_16K   (original baseline)
network=HybridBiMamba_SS_16K
config=config/inference/${network}.yaml

CUDA_VISIBLE_DEVICES=0 python3 -u inference.py --config $config
