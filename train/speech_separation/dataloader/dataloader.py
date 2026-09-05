"""
Dataloader for speech separation (adapted from ClearerVoice-Studio, which
itself is based on speechbrain).

Extension vs the original ClearerVoice-Studio dataloader: an optional 4th
column in the .scp files carries the mixture-level F0 contour (.npy, Hz,
10 ms frames, 0 = unvoiced) used by the F0-conditioned Hybrid Bi-Mamba model.
When ``use_f0`` is set and the scp line has 4 columns, each item returns
(inputs, labels, f0); otherwise it returns (inputs, labels) exactly like the
original framework (MossFormer2_SS etc. keep working unchanged).
"""

import numpy as np
import math
import os
import random
import torch
import torch.utils.data as data
import torch.distributed as dist
import soundfile as sf
from torch.utils.data import Dataset

from dataloader.misc import read_and_config_file
import librosa

EPS = 1e-6
F0_FRAME_PERIOD_S = 0.01  # 10 ms -> 100 fps


def audioread(path, sampling_rate):
    data, fs = sf.read(path)
    if fs != sampling_rate:
        data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
    if len(data.shape) > 1:
        data = data[:, 0]
    return data


def audioread_multi_wavs(paths, sampling_rate):
    data_list = []
    for path in paths:
        data, fs = sf.read(path)
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if len(data.shape) > 1:
            data = data[:, 0]
        data_list.append(data)
    max_len = 0
    for data in data_list:
        if max_len < len(data):
            max_len = len(data)
    data_list_padded = []
    for data in data_list:
        if max_len > len(data):
            data = zero_padding(data, max_len)
        data_list_padded.append(data)
    return np.transpose(np.asarray(data_list_padded))


def zero_padding(samples, desired_length):
    samples_to_add = desired_length - len(samples)
    if len(samples.shape) == 1:
        new_zeros = np.zeros(samples_to_add)
    elif len(samples.shape) == 2:
        new_zeros = np.zeros((samples_to_add, samples.shape[1]))
    return np.append(samples, new_zeros, axis=0)


def read_f0(f0_path, sr):
    """Read a mixture F0 contour (Hz, 100 fps); None when the file is absent."""
    if f0_path is None or not f0_path or not os.path.exists(f0_path):
        return None
    return np.load(f0_path).astype(np.float32)  # (n_frames,)


class DataReader(object):
    """Reads a list of wav files (for inference). Same as ClearerVoice-Studio."""

    def __init__(self, args):
        self.file_list = read_and_config_file(args.input_path, args, decode=True)
        self.sampling_rate = args.sampling_rate

    def extract_feature(self, path):
        utt_id = path.split("/")[-1]
        data = audioread(path, self.sampling_rate).astype(np.float32)
        inputs = np.reshape(data, [1, data.shape[0]])
        return inputs, utt_id, data.shape[0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])


class Wave_Processor(object):
    """Processes one mixture + N label wavs into equal-length segments.

    Random segment cropping (max_length seconds) is identical to the original;
    the F0 contour is cropped to the same segment (frame hop = 10 ms).
    """

    def process_multi_labels(self, path, segment_length, sampling_rate):
        try:
            wave_inputs = audioread(path["inputs"], sampling_rate)
            wave_labels = audioread_multi_wavs(path["labels"], sampling_rate)
        except Exception:
            print("audioread() failed, skip this batch!")
            return None, None, None
        max_len = max(wave_inputs.shape[0], wave_labels.shape[0])
        if max_len < segment_length:
            padded_inputs = zero_padding(wave_inputs, segment_length)
            padded_labels = zero_padding(wave_labels, segment_length)
            padded_inputs[: wave_inputs.shape[0]] = wave_inputs
            padded_labels[: wave_labels.shape[0], :] = wave_labels
            st_idx = 0
        else:
            st_idx = random.randint(0, max_len - segment_length)
            padded_inputs = wave_inputs[st_idx : st_idx + segment_length]
            padded_labels = wave_labels[st_idx : st_idx + segment_length, :]

        f0 = self._read_f0_segment(path, st_idx, segment_length, sampling_rate)
        return padded_inputs, padded_labels, f0

    @staticmethod
    def _read_f0_segment(path, st_idx, segment_length, sampling_rate):
        f0_path = path.get("f0")
        if not f0_path or not os.path.exists(f0_path):
            return None
        f0 = np.load(f0_path).astype(np.float32)  # (n_frames,) Hz @ 10 ms
        hop = int(sampling_rate * F0_FRAME_PERIOD_S)
        fr0 = st_idx // hop
        fr1 = int(np.ceil((st_idx + segment_length) / hop))
        seg = f0[fr0:fr1]
        if seg.shape[0] < 1:
            return None
        return seg


class AudioDataset(Dataset):
    def __init__(self, args, data_type):
        self.args = args
        self.sampling_rate = args.sampling_rate
        if data_type == "train":
            self.wav_list = read_and_config_file(args.tr_list, args)
        elif data_type == "val":
            self.wav_list = read_and_config_file(args.cv_list, args)
        elif data_type == "test":
            self.wav_list = read_and_config_file(args.tt_list, args)
        else:
            print("Data type: {} is unknown!".format(data_type))
        self.wav_processor = Wave_Processor()
        self.segment_length = self.sampling_rate * self.args.max_length
        self.need_f0 = bool(getattr(args, "use_f0", 0))
        if self.need_f0:
            n_before = len(self.wav_list)
            self.wav_list = [s for s in self.wav_list if s.get("f0")]
            if len(self.wav_list) != n_before:
                print(
                    "Warning: use_f0=1 but {} / {} scp lines have no F0 column -> dropped".format(
                        n_before - len(self.wav_list), n_before
                    )
                )
            if len(self.wav_list) == 0:
                raise RuntimeError(
                    "use_f0=1 requires a 4th column (mixture F0 .npy) in every scp line"
                )
        print("No. {} files: {}".format(data_type, len(self.wav_list)))

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        data_info = self.wav_list[index]
        while True:
            inputs, labels, f0 = self.wav_processor.process_multi_labels(
                data_info, self.segment_length, self.sampling_rate
            )
            if inputs is not None:
                break
            index = index + 1
            if index >= len(self.wav_list):
                index = random.randint(0, len(self.wav_list) - 1)
            data_info = self.wav_list[index]
        if self.need_f0 and f0 is not None:
            return inputs, labels, f0
        return inputs, labels


class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ind = (
                torch.randperm(int(len(self.dataset) / self.num_replicas), generator=g)
                * self.num_replicas
            )
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind + i).tolist()
        else:
            indices = list(range(len(self.dataset)))
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def collate_fn_2x_wavs(data):
    inputs, labels = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    return x, y


def collate_fn_3x_wavs_f0(data):
    inputs, labels, f0s = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    max_f0 = max(f.shape[0] for f in f0s)
    f0_mat = np.zeros((len(f0s), max_f0), dtype=np.float32)
    for e, f in enumerate(f0s):
        f0_mat[e, : f.shape[0]] = f
    z = torch.FloatTensor(f0_mat)
    return x, y, z


def get_dataloader(args, data_type):
    datasets = AudioDataset(args=args, data_type=data_type)

    sampler = (
        DistributedSampler(
            datasets,
            num_replicas=args.world_size,
            rank=args.local_rank,
        )
        if args.distributed
        else None
    )

    need_f0 = bool(getattr(args, "use_f0", 0))
    collate_fn = collate_fn_3x_wavs_f0 if need_f0 else collate_fn_2x_wavs

    generator = data.DataLoader(
        datasets,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        sampler=sampler,
    )
    return sampler, generator
