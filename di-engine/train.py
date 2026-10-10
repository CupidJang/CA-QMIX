"""
Train CA-QMIX or a baseline on a SMACv2 10-vs-10 scenario.

    python train.py --race protoss --method caqmix --seed 0
    python train.py --race zerg --method caqmix --ablation binary_gate
"""
import argparse

import torch
from ding.entry import serial_pipeline
from dizoo.smac2.config import get_config, RACES, METHODS, ABLATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--race', choices=RACES, default='protoss')
    parser.add_argument('--method', choices=METHODS, default='caqmix')
    parser.add_argument('--ablation', choices=ABLATIONS, help='CA-QMIX ablation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-env-step', type=float, default=1e7)
    parser.add_argument(
        '--obs-last-action',
        action=argparse.BooleanOptionalAction,
        help="include allies' last actions in the observation (default: as in the paper)"
    )
    parser.add_argument('--center-coop', action='store_true', help='subtract the action mean from Q_coop')
    parser.add_argument('--cpu', action='store_true', help='do not use the GPU even if one is available')
    args = parser.parse_args()

    main_config, create_config = get_config(
        race=args.race,
        method=args.method,
        seed=args.seed,
        ablation=args.ablation,
        obs_last_action=args.obs_last_action,
        center_coop=args.center_coop,
        cuda=torch.cuda.is_available() and not args.cpu,
    )
    serial_pipeline((main_config, create_config), seed=args.seed, max_env_step=int(args.max_env_step))


if __name__ == '__main__':
    main()
