import torch
import torch.nn as nn

EPS = 1e-8


class network_wrapper(nn.Module):
    """Registry of speech-separation networks (same role as the original
    ClearerVoice-Studio networks.py).

    Supported ``args.network`` values:
      * MossFormer2_SS_16K / MossFormer2_SS_8K  (original baseline, vendored)
      * HybridBiMamba_SS_16K / HybridBiMamba_SS_8K
        (Conformer-BiMamba; set ``use_f0: 1`` for Vietnamese F0 conditioning)
    """

    def __init__(self, args):
        super(network_wrapper, self).__init__()
        self.args = args
        if args.network in ["MossFormer2_SS_16K", "MossFormer2_SS_8K"]:
            from models.mossformer2.mossformer2 import MossFormer2_SS

            self.ss_network = MossFormer2_SS(args).model
        elif args.network in ["HybridBiMamba_SS_16K", "HybridBiMamba_SS_8K"]:
            from models.hybrid_bimamba.hybrid_bimamba import HybridBiMamba_SS

            self.ss_network = HybridBiMamba_SS(args).model
        else:
            print("in networks, {} is not found!".format(args.network))
            raise ValueError("unknown network: {}".format(args.network))

    def forward(self, mixture, f0=None):
        est_sources = self.ss_network(mixture, f0)
        return est_sources
