"""
Losses for training neural networks (condensed from ClearerVoice-Studio /
MossFormer2, which itself is based on speechbrain).

Provides the permutation-invariant negative SI-SNR loss used by the original
MossFormer2_SS pipeline. ``loss_hybrid_bimamba_ss`` is an alias: the F0
contour is consumed inside the Hybrid Bi-Mamba model, so the separation loss
has exactly the same form (PIT SI-SNR between the reference sources and the
list of estimated waveforms).
"""

import torch
import torch.nn as nn
from itertools import permutations


class PitWrapper(nn.Module):
    """Permutation Invariant Wrapper (speechbrain) — see original license."""

    def __init__(self, base_loss):
        super(PitWrapper, self).__init__()
        self.base_loss = base_loss

    def _fast_pit(self, loss_mat):
        loss = None
        assigned_perm = None
        for p in permutations(range(loss_mat.shape[0])):
            c_loss = loss_mat[range(loss_mat.shape[0]), p].mean()
            if loss is None or loss > c_loss:
                loss = c_loss
                assigned_perm = p
        return loss, assigned_perm

    def _opt_perm_loss(self, pred, target):
        n_sources = pred.size(-1)
        pred = pred.unsqueeze(-2).repeat(
            *[1 for x in range(len(pred.shape) - 1)], n_sources, 1
        )
        target = target.unsqueeze(-1).repeat(
            1, *[1 for x in range(len(target.shape) - 1)], n_sources
        )
        loss_mat = self.base_loss(pred, target)
        assert len(loss_mat.shape) >= 2, "Base loss should not reduce"
        mean_over = [x for x in range(len(loss_mat.shape))]
        loss_mat = loss_mat.mean(dim=mean_over[:-2])
        return self._fast_pit(loss_mat)

    def forward(self, preds, targets):
        losses = []
        perms = []
        for pred, label in zip(preds, targets):
            loss, p = self._opt_perm_loss(pred, label)
            perms.append(p)
            losses.append(loss)
        return torch.stack(losses), perms


def cal_si_snr(source, estimate_source):
    """Negative SI-SNR (minimisable). source/estimate: [T, B, C]."""
    EPS = 1e-8
    assert source.size() == estimate_source.size()
    source_lengths = torch.tensor(
        [estimate_source.shape[0]] * estimate_source.shape[1],
        device=estimate_source.device.type if estimate_source.is_cuda else "cpu",
    )
    mask = get_mask(source, source_lengths)
    estimate_source = estimate_source * mask

    num_samples = source_lengths.contiguous().reshape(1, -1, 1).float()
    mean_target = torch.sum(source, dim=0, keepdim=True) / num_samples
    mean_estimate = torch.sum(estimate_source, dim=0, keepdim=True) / num_samples
    s_target = source - mean_target
    s_estimate = estimate_source - mean_estimate
    s_target = s_target * mask
    s_estimate = s_estimate * mask

    dot = torch.sum(s_estimate * s_target, dim=0, keepdim=True)
    s_target_energy = torch.sum(s_target ** 2, dim=0, keepdim=True) + EPS
    proj = dot * s_target / s_target_energy
    e_noise = s_estimate - proj
    si_snr_beforelog = torch.sum(proj ** 2, dim=0) / (torch.sum(e_noise ** 2, dim=0) + EPS)
    si_snr = 10 * torch.log10(si_snr_beforelog + EPS)
    return -si_snr.unsqueeze(0)


def get_mask(source, source_lengths):
    """Mask used for padding — [T, B, 1]."""
    T, B, _ = source.size()
    mask = source.new_ones((T, B, 1))
    for i in range(B):
        mask[source_lengths[i] :, i, :] = 0
    return mask


def get_si_snr_with_pitwrapper(source, estimate_source):
    """PIT negative SI-SNR, one loss per utterance (speechbrain convention)."""
    pit_si_snr = PitWrapper(cal_si_snr)
    loss, _ = pit_si_snr(source, estimate_source)
    return loss


def loss_mossformer2_ss(args, inputs, labels, Out_List, device):
    """Loss for MossFormer2_SS / any time-domain separator.

    labels       : [B, T, C]  (reference sources)
    Out_List     : list of C tensors [B, T]  (estimated waveforms)
    Returns      : [B] negative SI-SNR per utterance (mean over speakers).
    """
    estimates = torch.stack(Out_List, dim=2)  # [B, T, C]
    loss = get_si_snr_with_pitwrapper(labels.to(device), estimates)
    return loss


def loss_hybrid_bimamba_ss(args, inputs, labels, Out_List, device, f0=None):
    """Loss for HybridBiMamba_SS. Identical form to MossFormer2's loss; the
    F0 contour is consumed inside the model and does not enter the loss."""
    return loss_mossformer2_ss(args, inputs, labels, Out_List, device)
