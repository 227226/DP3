import argparse
import os
from pathlib import Path

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',
                        default=os.getcwd(),
                        type=Path,
                        help='Directory with test patient subfolders')
    parser.add_argument('--json_train_path',
                        default='train_info.json',
                        type=Path,
                        help='Path to transformInfo json')
    parser.add_argument('--json_valid_path',
                        default='valid_info.json',
                        type=Path,
                        help='Path to transformInfo json')
    parser.add_argument('--json_test_path',
                        default='test_info.json',
                        type=Path,
                        help='Path to transformInfo json')
    parser.add_argument(
        '--n_classes',
        default=9,
        type=int,
        help='Number of predicted values')
    parser.add_argument('--model_depth',
                        default=34,
                        type=int,
                        help='Depth of ResNet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--num_epochs',
        default=50,
        type=int,
        help='Number of training epochs')
    parser.add_argument(
        '--learning_rate',
        default=0.0001,
        type=float,
        help='Learning rate')
    parser.add_argument(
        '--batch_size',
        default=4,
        type=int,
        help='Batch size')
    parser.add_argument(
        '--resize',
        default=1, # 0.3
        type=float,
        help='Resize input data isotropically by scale factor')
    args = parser.parse_args()

    return args