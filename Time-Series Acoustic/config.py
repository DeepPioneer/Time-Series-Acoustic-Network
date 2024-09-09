import torch
import numpy as np
import argparse

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
                        choices=['M18','Res1d','Res2d','WaveMsNet','Wavegram_Logmel','SimPFs', 'WTS','TSNA','Demo','original','TS','new_model'],
                        type=str, help='model choice') # 'Res1d'
    
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_folds',type=int,default=5)
    parser.add_argument('--n_epoch', type=int, default=60)
    parser.add_argument('--learning_rate', type=float, default=1e-3) # 1e-3
    parser.add_argument('--batch_size', type=int, default=128)
    
    parser.add_argument('--data_type', type=str, default='Cut_ShipEar', choices=['ESC', 'Cut_deepShip',"Cut_ShipEar","Cut_whale"])
    
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--window_size', type=int, default=1024)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--mel_bins', type=int, default=64)
    parser.add_argument('--fmin', type=int, default=50)
    parser.add_argument('--fmax', type=int, default=8000)

    parser.add_argument('--no_noise', type=str2bool, default=True)
    parser.add_argument('--noise_level', type=str, default='original')
    parser.add_argument('--noise_path', type=str, default="negativeFive_npz")# noise_negativeFifteen_npz   noise_negativeTen_npz
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--emb_dim', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--masking_ratio', type=float, default=0.4)
    parser.add_argument('--dropout_rate', type=float, default=0.15)
    parser.add_argument('--patch_size', type=int, default=8)

    # TSLANet components:
    parser.add_argument('--ICB', type=str2bool, default=True)
    parser.add_argument('--ASB', type=str2bool, default=True)
    parser.add_argument('--adaptive_filter', type=str2bool, default=True)
    
       # Optimization parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"') # SimPFs 2d SGD 
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    return parser

