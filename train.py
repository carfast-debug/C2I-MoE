"""Command-line entry point for C2I-MoE training."""

import argparse
import json
import os
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1-root', type=Path, required=True)
    parser.add_argument('--fmri-root', type=Path, required=True)
    parser.add_argument('--t1-pretrained', type=Path, required=True)
    parser.add_argument('--fmri-pretrained', type=Path, required=True)
    parser.add_argument('--init-checkpoint', type=Path, help='Optional complete C2I-MoE state dict for fine-tuning.')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=28)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--interaction-weight', type=float, default=0.5)
    parser.add_argument('--contrast-weight', type=float, default=0.02)
    parser.add_argument('--contrast-temperature', type=float, default=0.10)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ['t1_root', 'fmri_root', 't1_pretrained', 'fmri_pretrained', 'init_checkpoint', 'output_dir']:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    for name in ['t1_root', 'fmri_root']:
        if not getattr(args, name).is_dir():
            parser.error(f'{name} is not a directory: {getattr(args, name)}')
    for name in ['t1_pretrained', 'fmri_pretrained', 'init_checkpoint']:
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f'{name} is not a file: {value}')
    if args.batch_size < 2 or args.epochs < 1 or args.num_workers < 0:
        parser.error('batch-size must be >=2, epochs >=1, and num-workers >=0.')
    if args.contrast_temperature <= 0 or args.contrast_weight < 0 or args.interaction_weight < 0:
        parser.error('Loss weights must be nonnegative and contrast temperature positive.')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('Use a new or empty output directory to avoid overwriting an experiment.')

    import torch
    import training

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    training.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    training.T1_ROOT, training.FMRI_ROOT = str(args.t1_root), str(args.fmri_root)
    training.T1_PRETRAIN, training.FMRI_PRETRAIN = str(args.t1_pretrained), str(args.fmri_pretrained)
    training.INIT_CHECKPOINT = str(args.init_checkpoint) if args.init_checkpoint else None
    training.BATCH_SIZE, training.NUM_EPOCHS = args.batch_size, args.epochs
    training.NUM_WORKERS = args.num_workers
    training.CONTRAST_LOSS_WEIGHT = args.contrast_weight
    training.CONTRAST_TEMPERATURE = args.contrast_temperature

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration['selection_score'] = training.SELECTION_SCORE
    (args.output_dir / 'train_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    previous_dir = Path.cwd()
    try:
        os.chdir(args.output_dir)
        return training.run_experiment(args.split_seed, args.seed, args.interaction_weight)
    finally:
        os.chdir(previous_dir)


if __name__ == '__main__':
    main()
