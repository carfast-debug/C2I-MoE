"""Evaluate one C2I-MoE checkpoint on an explicit held-out split."""

import argparse
import json
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--t1-root', type=Path, required=True)
    parser.add_argument('--fmri-root', type=Path, required=True)
    parser.add_argument('--layout', choices=['private', 'ppmi'], default='private')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--split-csv', type=Path, help='CSV emitted by training; columns: subject_id,set.')
    selection.add_argument('--all-subjects', action='store_true', help='Explicitly evaluate every supplied paired sample.')
    parser.add_argument('--partition', choices=['train', 'val', 'test'], default='test')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, required=True, help='Write aggregate metrics as JSON (0-1 units).')
    parser.add_argument('--allow-missing-contrast-heads', action='store_true', help='Only for legacy checkpoints lacking training-only projection heads.')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ['checkpoint', 'split_csv']:
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f'{name} is not a file: {value}')
    for name in ['t1_root', 'fmri_root']:
        if not getattr(args, name).is_dir():
            parser.error(f'{name} is not a directory: {getattr(args, name)}')
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error('batch-size must be positive and num-workers nonnegative.')
    if args.output.exists():
        parser.error('Choose a new output JSON file to preserve previous evaluations.')

    import torch
    from torch.utils.data import DataLoader
    from checkpoint_utils import load_checkpoint
    from evaluation import classification_metrics, select_partition
    from fusion_model import FMRIFeatureExtractor, I2MoE_Fusion, T1FeatureExtractor
    from paired_dataset import PairedDataset, create_paired_samples

    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if args.layout == 'ppmi':
        from ppmi_dataset import PPMIDataset
        dataset = PPMIDataset(str(args.fmri_root), str(args.t1_root))
    else:
        dataset = PairedDataset(create_paired_samples(str(args.fmri_root), str(args.t1_root)), augment=False)
    if args.split_csv:
        dataset.samples = select_partition(dataset.samples, args.split_csv, args.partition, ppmi=args.layout == 'ppmi')
    if not len(dataset):
        raise ValueError('No paired samples were found. Check the documented filename layout.')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    # Complete training checkpoints contain both foundation encoders; no extra
    # pretrained-weight files are needed for inference.
    model = I2MoE_Fusion(T1FeatureExtractor(), FMRIFeatureExtractor(), hidden_size=768,
                        num_classes=2, dropout=0.5, interaction_loss_weight=0.5,
                        temperature=2, num_layers=3, gated_layers=[2],
                        contrast_dim=128, contrast_temperature=0.10)
    load_checkpoint(model, args.checkpoint, args.allow_missing_contrast_heads)
    model.to(device).eval()
    labels, probabilities = [], []
    with torch.inference_mode():
        for batch in loader:
            t1, fmri, adjacency, targets = batch[:4]
            logits = model(t1.to(device), fmri.to(device), adjacency.to(device), parc_type='aal116', disease_type='PD')
            probabilities.extend(torch.softmax(logits.float(), dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    result = classification_metrics(labels, probabilities)
    result.update(layout=args.layout, partition=args.partition if args.split_csv else 'all-subjects')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == '__main__':
    main()
