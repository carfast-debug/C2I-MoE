"""Load one C2I-MoE state dictionary without silently skipping fusion weights."""


def load_checkpoint(model, path, allow_missing_contrast_heads=False):
    import torch

    state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict):
        raise ValueError('Expected a state dictionary or a dictionary containing state_dict.')
    if 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict) or not state:
        raise ValueError('The checkpoint state dictionary is empty or invalid.')
    if all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    if not allow_missing_contrast_heads:
        model.load_state_dict(state, strict=True)
        return
    missing, unexpected = model.load_state_dict(state, strict=False)
    forbidden = [key for key in missing if not key.startswith(('t1_contrast_head.', 'fmri_contrast_head.'))]
    if forbidden or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: missing={forbidden}, unexpected={unexpected}')
