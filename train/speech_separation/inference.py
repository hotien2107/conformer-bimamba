import torch
import os

from utils.misc import reload_for_eval
from utils.decode import decode_one_audio
from dataloader.dataloader import DataReader
import yamlargparse
import soundfile as sf
import warnings
from networks import network_wrapper

warnings.filterwarnings("ignore")


def inference(args):
    device = torch.device("cuda") if args.use_cuda == 1 else torch.device("cpu")
    print(device)
    print("creating model...")
    model = network_wrapper(args).ss_network
    model.to(device)

    print("loading model ...")
    reload_for_eval(model, args.checkpoint_dir, args.use_cuda)
    model.eval()
    with torch.no_grad():
        data_reader = DataReader(args)
        output_wave_dir = args.output_dir
        if not os.path.isdir(output_wave_dir):
            os.makedirs(output_wave_dir)
        num_samples = len(data_reader)
        print("Decoding...")
        for idx in range(num_samples):
            input_audio, wav_id, input_len = data_reader[idx]
            print("audio: {}".format(wav_id))
            output_audios = decode_one_audio(model, device, input_audio, args)
            for spk in range(args.num_spks):
                output_audio = output_audios[spk][:input_len]
                sf.write(
                    os.path.join(output_wave_dir, wav_id.replace(".wav", "_s{}.wav".format(spk + 1))),
                    output_audio,
                    args.sampling_rate,
                )
    print("Done!")


if __name__ == "__main__":
    parser = yamlargparse.ArgumentParser("Settings")
    parser.add_argument("--config", help="config file path", action=yamlargparse.ActionConfigFile)

    parser.add_argument("--mode", type=str, default="inference", help="run train or inference")
    parser.add_argument("--checkpoint-dir", dest="checkpoint_dir", type=str,
                        default="checkpoints/HybridBiMamba_SS_16K", help="the checkpoint dir")
    parser.add_argument("--input-path", dest="input_path", type=str,
                        help="input dir or scp file for saving noisy audio")
    parser.add_argument("--output-dir", dest="output_dir", type=str,
                        help="output dir for saving processed audio")
    parser.add_argument("--use-cuda", dest="use_cuda", default=1, type=int, help="use cuda")
    parser.add_argument("--num-gpu", dest="num_gpu", type=int, default=1, help="the num gpus to use")
    parser.add_argument("--network", type=str,
                        help="select speech separation models: MossFormer2_SS_16K, "
                             "MossFormer2_SS_8K, HybridBiMamba_SS_16K, HybridBiMamba_SS_8K")
    parser.add_argument("--sampling-rate", dest="sampling_rate", type=int, default=16000)

    # model params
    parser.add_argument("--num-spks", dest="num_spks", type=int, default=2)
    parser.add_argument("--one-time-decode-length", dest="one_time_decode_length", type=int,
                        default=60, help="the max length (second) for one-time pass decoding")
    parser.add_argument("--decode-window", dest="decode_window", type=int, default=1,
                        help="segmental decoding window length (second)")
    parser.add_argument("--encoder_kernel-size", dest="encoder_kernel_size", type=int, default=16)
    parser.add_argument("--encoder-embedding-dim", dest="encoder_embedding_dim", type=int, default=512)
    parser.add_argument("--mossformer-squence-dim", dest="mossformer_sequence_dim", type=int, default=512)
    parser.add_argument("--num-mossformer_layer", dest="num_mossformer_layer", type=int, default=24)

    # HybridBiMamba params (kept consistent with train.py)
    parser.add_argument("--chunk-size", dest="chunk_size", type=int, default=250)
    parser.add_argument("--num-intra", dest="num_intra", type=int, default=4)
    parser.add_argument("--num-inter", dest="num_inter", type=int, default=4)
    parser.add_argument("--d-ffn", dest="d_ffn", type=int, default=1024)
    parser.add_argument("--conv-kernel", dest="conv_kernel", type=int, default=31)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d-state", dest="d_state", type=int, default=64)
    parser.add_argument("--d-conv", dest="d_conv", type=int, default=4)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--combine", type=str, default="sum")
    parser.add_argument("--use-mamba2", dest="use_mamba2", type=int, default=0)
    parser.add_argument("--use-f0", dest="use_f0", type=int, default=0)
    parser.add_argument("--f0-dim", dest="f0_dim", type=int, default=64)
    parser.add_argument("--checkpointing", type=int, default=0)

    args = parser.parse_args()
    print(args)
    inference(args)
