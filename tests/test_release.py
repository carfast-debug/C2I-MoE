import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from checkpoint_utils import load_checkpoint
from evaluation import classification_metrics, select_partition
from fusion_model import I2MoE_Fusion


class TinyT1(nn.Module):
    def forward(self, x):
        return x


class TinyFMRI(nn.Module):
    def forward(self, x, adjacency, parc_type, disease_type):
        return x


def tiny_model():
    return I2MoE_Fusion(TinyT1(), TinyFMRI(), hidden_size=12, num_layers=1,
                       dropout=0.0, gated_layers=[0], contrast_dim=8)


def test_fusion_gradients_and_checkpoint_rotation_roundtrip(tmp_path):
    torch.manual_seed(13)
    model = tiny_model().train()
    t1, fmri, adjacency = torch.randn(3, 1024), torch.randn(3, 256), torch.zeros(3, 1)
    logits, interaction_losses, contrast = model(t1, fmri, adjacency, return_contrast_loss=True)
    loss = logits.square().mean() + sum(interaction_losses) / len(interaction_losses) + 0.02 * contrast
    assert torch.isfinite(loss)
    loss.backward()
    assert model.proj_t1.weight.grad is not None
    assert model.t1_contrast_head[-1].weight.grad is not None
    assert model.fmri_contrast_head[-1].weight.grad is not None
    rotation = model.i2moe.interaction_experts[0].fusion_model.rot_T1
    rotation.copy_(torch.roll(torch.eye(12), shifts=2, dims=0))
    model.eval()
    expected = model(t1, fmri, adjacency)
    checkpoint = tmp_path / 'model.pth'
    torch.save(model.state_dict(), checkpoint)
    restored = tiny_model().eval()
    load_checkpoint(restored, checkpoint)
    torch.testing.assert_close(restored(t1, fmri, adjacency), expected)
    torch.testing.assert_close(restored.i2moe.interaction_experts[0].fusion_model.rot_T1, rotation)


def test_checkpoint_rejects_missing_fusion_weight(tmp_path):
    model = tiny_model()
    state = model.state_dict()
    del state['proj_t1.weight']
    path = tmp_path / 'broken.pth'
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(model, path, allow_missing_contrast_heads=True)


def test_missing_auxiliary_heads_are_explicitly_optional(tmp_path):
    model = tiny_model()
    state = {key: value for key, value in model.state_dict().items()
             if not key.startswith(('t1_contrast_head.', 'fmri_contrast_head.'))}
    path = tmp_path / 'legacy.pth'
    torch.save(state, path)
    with pytest.raises(RuntimeError):
        load_checkpoint(tiny_model(), path)
    load_checkpoint(tiny_model(), path, allow_missing_contrast_heads=True)


def test_metrics_positive_label_and_single_class_handling():
    probabilities = [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.7, 0.3]]
    metrics = classification_metrics([0, 0, 1, 1], probabilities)
    assert metrics['confusion_matrix'] == [[1, 1], [1, 1]]
    assert metrics['accuracy'] == metrics['sensitivity'] == metrics['specificity'] == 0.5
    assert metrics['auc'] == pytest.approx(0.75)
    assert classification_metrics([1], [[0.2, 0.8]])['auc'] is None


def test_split_selection_groups_visits_and_rejects_leakage(tmp_path):
    split = tmp_path / 'split.csv'
    split.write_text('subject_id,set\nPD001,test\nNC001,train\n', encoding='utf-8')
    samples = [('/fake/m0wrp1PD001_T1w.nii', 'a.mat', 1),
               ('/fake/m0wrp1PD001_12m_T1w.nii', 'b.mat', 1),
               ('/fake/m0wrp1NC001_T1w.nii', 'c.mat', 0)]
    assert len(select_partition(samples, split, 'test')) == 2
    split.write_text('subject_id,set\nPD001,test\nPD001,train\n', encoding='utf-8')
    with pytest.raises(ValueError, match='multiple partitions'):
        select_partition(samples, split, 'test')


def test_private_reader_with_synthetic_nifti_and_mat(tmp_path):
    import nibabel as nib
    from scipy.io import savemat
    from paired_dataset import PairedDataset, create_paired_samples

    t1_dir = tmp_path / 'T1' / 'PD'
    fmri_dir = tmp_path / 'fMRI' / 'PD' / 'sub-PD001'
    t1_dir.mkdir(parents=True)
    fmri_dir.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.zeros((112, 112, 112), dtype=np.float32), np.eye(4)),
             t1_dir / 'm0wrp1PD001_T1w.nii')
    savemat(fmri_dir / 'sub-PD001_Classic_results.mat', {'corr_matrix': np.eye(116)})
    pairs = create_paired_samples(str(tmp_path / 'fMRI'), str(tmp_path / 'T1'))
    assert len(pairs) == 1
    t1, correlation, adjacency, label = PairedDataset(pairs)[0]
    assert t1.shape == (1, 112, 112, 112)
    assert correlation.shape == adjacency.shape == (116, 116)
    assert label.item() == 1


@pytest.mark.parametrize('seed_args, expected_seeds', [
    ([], (42, 42)),
    (['--split-seed', '17', '--seed', '123'], (17, 123)),
])
def test_train_cli_forwards_one_run_and_restores_working_directory(tmp_path, monkeypatch, seed_args, expected_seeds):
    import training
    from train import main

    t1_dir, fmri_dir = tmp_path / 'T1', tmp_path / 'fMRI'
    t1_dir.mkdir()
    fmri_dir.mkdir()
    t1_weights, fmri_weights = tmp_path / 't1.pth', tmp_path / 'fmri.pth'
    t1_weights.touch()
    fmri_weights.touch()
    captured = []
    monkeypatch.setattr(training, 'run_experiment', lambda split, seed, weight: captured.append((split, seed, weight)))
    previous = Path.cwd()
    output = tmp_path / 'out'
    main(['--t1-root', str(t1_dir), '--fmri-root', str(fmri_dir),
          '--t1-pretrained', str(t1_weights), '--fmri-pretrained', str(fmri_weights),
          '--output-dir', str(output),
          '--num-workers', '0', '--device', 'cpu'] + seed_args)
    assert captured == [(*expected_seeds, 0.5)]
    assert training.SELECTION_SCORE == 'auc-f1'
    assert (output / 'train_config.json').is_file()
    config = json.loads((output / 'train_config.json').read_text())
    assert config['selection_score'] == 'auc-f1'
    assert (config['split_seed'], config['seed']) == expected_seeds
    assert Path.cwd() == previous
