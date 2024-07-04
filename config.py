import torch
import numpy as np
import argparse

torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)


def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Model Training', add_help=False)

    parser.add_argument('--model_name', default='WaveMsNet',
                        choices=['WaveMsNet', 'SincNet', 'SimPFs',
                                 'Wavegram_Logmel_Cnn', 'Wavegram_Logmel128_Cnn', 'audio_fcanet', 'my_model', 'WTS',
                                 'DFT', 'DWT', 'PTT', 'TSNA'],
                        type=str, help='model choice')

    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_folds', type=int, default=5)
    parser.add_argument('--n_epoch', type=int, default=60)  # 300
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--data_type', type=str, default='ESC', choices=['ESC', 'Deepship', "ShipsEar", "Whale"])

    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=14000)

    parser.add_argument('--no_noise', type=str2bool, default=True)
    parser.add_argument('--noise_level', type=str, default='original')
    parser.add_argument('--noise_path', type=str,
                        default="noise_negativeFive_npz")  # noise_negativeFifteen_npz   noise_negativeTen_npz
    # parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # TSLANet components:
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)

    return parser

