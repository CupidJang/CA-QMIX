"""
Collation of sequence samples for the recurrent multi-agent policies (QMIX, QTRAN, CollaQ, CA-QMIX).

``ding.utils.data.timestep_collate`` assumes equal-length samples and stacks ``prev_state`` before
removing it, which fails for RNN policies whose ``prev_state`` is a list of per-agent ``{h, c}``
dicts and for the nested SMAC observation dict. Samples of different lengths are zero-padded and
a ``(T, B)`` mask removes the padding from the loss.
"""

from typing import Any, Dict, List, Optional, Tuple
import copy

import torch

from easydict import EasyDict

from ding.torch_utils import to_device
from ding.torch_utils.network import sequence_mask
from ding.utils.data import default_collate

__all__ = ['pad_episodes', 'collate_episodes']


def _make_padding_step(template: Dict[str, Any]) -> Dict[str, Any]:
    """Zero timestep used for padding; ``done=1`` so that it is harmless even if it were not masked."""

    def _zero(v):
        if isinstance(v, torch.Tensor):
            return torch.zeros_like(v)
        if isinstance(v, dict):
            return {k: _zero(x) for k, x in v.items()}
        return copy.deepcopy(v)

    step = {k: _zero(v) for k, v in template.items()}
    if isinstance(step.get('done'), torch.Tensor):
        step['done'] = torch.ones_like(step['done'])
    else:
        step['done'] = True
    return step


def pad_episodes(
        episodes: List[List[Dict[str, Any]]]
) -> Tuple[List[List[Dict[str, Any]]], torch.Tensor]:
    """Zero-pads the samples to the longest one and returns them with a ``(T, B)`` validity mask."""
    lengths = [len(ep) for ep in episodes]
    max_len = max(lengths)
    padded = []
    for ep in episodes:
        n_pad = max_len - len(ep)
        if n_pad > 0:
            # The padding steps are identical and never modified afterwards, so one step is shared.
            ep = list(ep) + [_make_padding_step(ep[-1])] * n_pad
        padded.append(ep)
    # sequence_mask gives (B, T); the losses are (T, B).
    mask = sequence_mask(torch.as_tensor(lengths), max_len=max_len).t().float()
    return padded, mask


def _collate_timesteps(batch: List[List[Dict[str, Any]]]) -> EasyDict:
    """Stacks a batch of samples into ``{key: Tensor(T, B, ...)}`` (``prev_state`` is kept as a list)."""
    T = len(batch[0])
    prev_state = [[episode[t].get('prev_state') for episode in batch] for t in range(T)]
    drop = ('prev_state', 'collect_iter')
    per_timestep = []
    for t in range(T):
        step = default_collate([{k: v for k, v in episode[t].items() if k not in drop}
                                for episode in batch])  # {key: (B, ...)}
        _squeeze_1dim(step)
        per_timestep.append(step)
    data = EasyDict(default_collate(per_timestep))  # {key: (T, B, ...)}
    data['prev_state'] = prev_state
    return data


def _squeeze_1dim(node: Any) -> None:
    """Squeezes ``(B, 1)`` tensors to ``(B,)`` as ``timestep_collate`` does.

    Otherwise ``reward`` and ``done`` stay ``(T, B, 1)`` and silently broadcast against
    ``total_q`` ``(T, B)`` in the TD loss.
    """
    for k in list(node.keys()):
        v = node[k]
        if isinstance(v, torch.Tensor):
            if v.dim() == 2 and v.shape[1] == 1:
                node[k] = v.squeeze(-1)
        elif isinstance(v, dict):
            _squeeze_1dim(v)


def collate_episodes(
        data: List[Any],
        cuda: bool = False,
        device: str = 'cpu',
) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
    """
    Returns:
        - data: ``{key: Tensor(T, B, ...)}``
        - episode_mask: ``(T, B)`` validity mask, or ``None`` if all samples have the same length.
    """
    episode_mask = None
    episode_input = bool(data) and isinstance(data[0], (list, tuple))
    if episode_input:
        if len({len(episode) for episode in data}) > 1:
            data, episode_mask = pad_episodes(data)
        data = _collate_timesteps(data)
    else:
        from ding.utils.data import timestep_collate
        data = timestep_collate(data)
    if cuda:
        data = to_device(data, device)
        if episode_mask is not None:
            episode_mask = to_device(episode_mask, device)
    return data, episode_mask
