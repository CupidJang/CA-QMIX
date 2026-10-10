"""QPLEX (DMAQ): Duplex Dueling Multi-Agent Q-Learning.

Ported from hijkzzz/pymarl2 @ 8ccac7c5aa134422a2e3009be735d23cdf8ce2f8
        src/modules/mixers/dmaq_general.py, dmaq_si_weight.py
        (originally wjh720/QPLEX; Wang et al., ICLR 2021)

Notes on the DI-engine port
  - QPLEX splits Q_i into V_i + A_i, so the mixer also needs each agent's max Q and the chosen action.
    The DI-engine ``Mixer`` signature (agent_qs, states) does not carry them, so ``QMix.forward`` is
    overridden to build and pass both.
  - A flat global state reaches the mixer unchanged through the Identity encoder (as in QMix).
  - The agent network (DRQN) is the same as in QMIX, so only the mixer differs.
"""
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.utils import MODEL_REGISTRY, list_split
from ding.torch_utils import to_tensor
from functools import reduce
from .qmix import QMix


class DMAQSIWeight(nn.Module):
    """Overview: multi-head, state- and action-dependent weights for the advantage term."""

    def __init__(
            self, agent_num: int, action_shape: int, state_dim: int,
            num_kernel: int = 4, adv_hypernet_embed: int = 64
    ) -> None:
        super(DMAQSIWeight, self).__init__()
        self.n_agents = agent_num
        self.n_actions = action_shape
        self.state_dim = state_dim
        self.action_dim = agent_num * action_shape
        self.state_action_dim = state_dim + self.action_dim
        self.num_kernel = num_kernel

        def _head(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, adv_hypernet_embed), nn.ReLU(), nn.Linear(adv_hypernet_embed, out_dim)
            )

        self.key_extractors = nn.ModuleList([_head(state_dim, 1) for _ in range(num_kernel)])
        self.agents_extractors = nn.ModuleList([_head(state_dim, agent_num) for _ in range(num_kernel)])
        self.action_extractors = nn.ModuleList([_head(self.state_action_dim, agent_num) for _ in range(num_kernel)])

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = torch.cat([states, actions], dim=1)

        weights = []
        for k_ext, a_ext, s_ext in zip(self.key_extractors, self.agents_extractors, self.action_extractors):
            x_key = torch.abs(k_ext(states)).repeat(1, self.n_agents) + 1e-10
            x_agents = torch.sigmoid(a_ext(states))
            x_action = torch.sigmoid(s_ext(data))
            weights.append(x_key * x_agents * x_action)
        return torch.stack(weights, dim=1).view(-1, self.num_kernel, self.n_agents).sum(dim=1)


class DMAQer(nn.Module):
    """Overview: QPLEX mixer. The V term is a plain sum, the A term a state- and action-dependent weighted sum."""

    def __init__(
            self, agent_num: int, action_shape: int, state_dim: int,
            mixing_embed_dim: int = 32, hypernet_embed: int = 64,
            num_kernel: int = 4, adv_hypernet_embed: int = 64,
            weighted_head: bool = True, is_minus_one: bool = True
    ) -> None:
        super(DMAQer, self).__init__()
        self.n_agents = agent_num
        self.n_actions = action_shape
        self.state_dim = state_dim
        self.action_dim = agent_num * action_shape
        self.embed_dim = mixing_embed_dim
        self.weighted_head = weighted_head
        self.is_minus_one = is_minus_one

        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, agent_num)
        )
        self.V = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed), nn.ReLU(), nn.Linear(hypernet_embed, agent_num)
        )
        self.si_weight = DMAQSIWeight(agent_num, action_shape, state_dim, num_kernel, adv_hypernet_embed)

    def forward(
            self, agent_qs: torch.Tensor, states: torch.Tensor,
            actions: torch.Tensor = None, max_q_i: torch.Tensor = None, is_v: bool = False
    ) -> torch.Tensor:
        leading = agent_qs.shape[:-1]
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)

        w_final = torch.abs(self.hyper_w_final(states)).reshape(-1, self.n_agents) + 1e-10
        v = self.V(states).reshape(-1, self.n_agents)
        if self.weighted_head:
            agent_qs = w_final * agent_qs + v

        if is_v:
            y = agent_qs.sum(dim=-1)
        else:
            max_q_i = max_q_i.reshape(-1, self.n_agents)
            if self.weighted_head:
                max_q_i = w_final * max_q_i + v
            adv_q = (agent_qs - max_q_i).detach()
            adv_w = self.si_weight(states, actions).reshape(-1, self.n_agents)
            y = (adv_q * (adv_w - 1.)).sum(dim=1) if self.is_minus_one else (adv_q * adv_w).sum(dim=1)
        return y.reshape(*leading)


@MODEL_REGISTRY.register('qplex')
class QPLEX(QMix):
    """Overview:
        QMIX with the mixer replaced by DMAQ, which computes the V term (is_v=True) and the A term
        separately and adds them.
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: Union[int, List[int]],
            action_shape: int,
            hidden_size_list: list,
            mixer: bool = True,
            lstm_type: str = 'gru',
            activation: nn.Module = nn.ReLU(),
            dueling: bool = False,
            mixing_embed_dim: int = 32,
            num_kernel: int = 4,
    ) -> None:
        super(QPLEX, self).__init__(
            agent_num, obs_shape, global_obs_shape, action_shape, hidden_size_list,
            mixer=mixer, lstm_type=lstm_type, activation=activation, dueling=dueling
        )
        self._action_shape = action_shape
        self._agent_num = agent_num
        if self.mixer:
            assert isinstance(global_obs_shape, int), "QPLEX only supports a flat global state"
            self._mixer = DMAQer(
                agent_num, action_shape, global_obs_shape,
                mixing_embed_dim=mixing_embed_dim, num_kernel=num_kernel
            )
            self._global_state_encoder = nn.Identity()

    def forward(self, data: dict, single_step: bool = True, need_mixer: bool = True) -> dict:
        """Same as QMix.forward, but also passes max_q_i and the one-hot action to the mixer."""
        agent_state, global_state, prev_state = data['obs']['agent_state'], data['obs']['global_state'], data[
            'prev_state']
        action = data.get('action', None)
        if single_step:
            agent_state, global_state = agent_state.unsqueeze(0), global_state.unsqueeze(0)
        if single_step and len(global_state.shape) == 2:
            global_state = global_state.unsqueeze(0)
        T, B, A = agent_state.shape[:3]

        assert len(prev_state) == B and all([len(p) == A for p in prev_state]), '{}-{}-{}-{}'.format(
            [type(p) for p in prev_state], B, A, len(prev_state[0])
        )
        prev_state = reduce(lambda x, y: x + y, prev_state)
        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        output = self._q_network({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        agent_q, next_state = output['logit'], output['next_state']
        next_state, _ = list_split(next_state, step=A)
        agent_q = agent_q.reshape(T, B, A, -1)

        action_mask = data['obs']['action_mask']
        if len(action_mask.shape) == 3:
            action_mask = action_mask.unsqueeze(0)
        if action is None:
            agent_q[action_mask == 0.0] = -9999999
            action = agent_q.argmax(dim=-1)
        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)   # (T,B,A)

        # need_mixer=False: the action-selection forward of double Q only needs the logits
        if self.mixer and need_mixer:
            global_state_embedding = self._process_global_state(global_state)
            # Extra mixer inputs: each agent's max Q and the one-hot chosen action
            masked_q = agent_q.masked_fill(action_mask == 0.0, -9999999)
            max_q_i = masked_q.max(dim=-1)[0]                                                 # (T,B,A)
            action_onehot = F.one_hot(action, num_classes=self._action_shape).float()         # (T,B,A,act)
            v_tot = self._mixer(agent_q_act, global_state_embedding, is_v=True)
            a_tot = self._mixer(
                agent_q_act, global_state_embedding, actions=action_onehot, max_q_i=max_q_i, is_v=False
            )
            total_q = v_tot + a_tot
        else:
            total_q = agent_q_act.sum(dim=-1)

        if single_step:
            total_q, agent_q = total_q.squeeze(0), agent_q.squeeze(0)
        return {'total_q': total_q, 'logit': agent_q, 'next_state': next_state, 'action_mask': data['obs']['action_mask']}
