"""Qatten: Attention-based mixing network.

Ported from hijkzzz/pymarl2 @ 8ccac7c5aa134422a2e3009be735d23cdf8ce2f8
        src/modules/mixers/qatten.py  (Yang et al., 2020)

Notes on the DI-engine port
  - The Qatten mixer's forward(agent_qs, states) has the same signature as the DI-engine ``Mixer``. With a
    flat global state (as in SMAC), QMix uses ``_global_state_encoder = nn.Identity()``, so the raw global
    state reaches the mixer and the mixer is a drop-in replacement.
  - pymarl2 reads its settings from an ``args`` object; here they are explicit arguments.
  - ``agent_own_state_size`` is the size of one agent's block in the global state. The SMAC state is
    [ally 1..n | enemy 1..m | last_action | timestep]; Qatten uses the ally blocks as keys.
"""
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ding.utils import MODEL_REGISTRY
from .qmix import QMix


class QattenMixer(nn.Module):
    """Overview:
        Mixes the agent Q-values with per-agent weights from multi-head attention over the state.
    """

    def __init__(
            self,
            agent_num: int,
            state_dim: int,
            agent_own_state_size: int,
            n_query_embedding_layer1: int = 64,
            n_query_embedding_layer2: int = 32,
            n_key_embedding_layer1: int = 32,
            n_head_embedding_layer1: int = 64,
            n_head_embedding_layer2: int = 4,
            n_attention_head: int = 4,
            n_constrant_value: int = 32,
            weighted_head: bool = True,
    ) -> None:
        super(QattenMixer, self).__init__()
        self.n_agents = agent_num
        self.state_dim = int(np.prod(state_dim)) if not isinstance(state_dim, int) else state_dim
        self.u_dim = agent_own_state_size
        self.n_attention_head = n_attention_head
        self.n_query_embedding_layer2 = n_query_embedding_layer2
        self.n_key_embedding_layer1 = n_key_embedding_layer1
        self.n_head_embedding_layer2 = n_head_embedding_layer2
        self.weighted_head = weighted_head

        self.query_embedding_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.state_dim, n_query_embedding_layer1), nn.ReLU(),
                    nn.Linear(n_query_embedding_layer1, n_query_embedding_layer2)
                ) for _ in range(n_attention_head)
            ]
        )
        self.key_embedding_layers = nn.ModuleList(
            [nn.Linear(self.u_dim, n_key_embedding_layer1) for _ in range(n_attention_head)]
        )
        self.scaled_product_value = np.sqrt(n_query_embedding_layer2)

        self.head_embedding_layer = nn.Sequential(
            nn.Linear(self.state_dim, n_head_embedding_layer1), nn.ReLU(),
            nn.Linear(n_head_embedding_layer1, n_head_embedding_layer2)
        )
        self.constrant_value_layer = nn.Sequential(
            nn.Linear(self.state_dim, n_constrant_value), nn.ReLU(), nn.Linear(n_constrant_value, 1)
        )

    def _get_us(self, states: torch.Tensor) -> torch.Tensor:
        """Per-agent own-state blocks from the front of the global state."""
        with torch.no_grad():
            us = states[:, :self.u_dim * self.n_agents].reshape(-1, self.u_dim)
        return us

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """Shapes:
            - agent_qs: :math:`(T, B, A)`   - states: :math:`(T, B, M)`   - return: :math:`(T, B)`
        """
        leading = agent_qs.shape[:-1]                      # (T, B)
        states = states.reshape(-1, self.state_dim)
        us = self._get_us(states)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](states).reshape(-1, 1, self.n_query_embedding_layer2)
            u_embedding = self.key_embedding_layers[i](us)
            u_embedding = u_embedding.reshape(-1, self.n_agents, self.n_key_embedding_layer1).permute(0, 2, 1)
            raw_lambda = torch.matmul(state_embedding, u_embedding) / self.scaled_product_value
            q_lambda_list.append(F.softmax(raw_lambda, dim=-1))

        q_lambda_list = torch.stack(q_lambda_list, dim=1).squeeze(-2)      # (-1, head, A)
        q_lambda_list = q_lambda_list.permute(0, 2, 1)                     # (-1, A, head)
        q_h = torch.matmul(agent_qs, q_lambda_list)                        # (-1, 1, head)

        if self.weighted_head:
            w_h = torch.abs(self.head_embedding_layer(states)).reshape(-1, self.n_head_embedding_layer2, 1)
            sum_q_h = torch.matmul(q_h, w_h).reshape(-1, 1)
        else:
            sum_q_h = q_h.sum(-1).reshape(-1, 1)

        q_tot = sum_q_h + self.constrant_value_layer(states)
        return q_tot.reshape(*leading)


@MODEL_REGISTRY.register('qatten')
class Qatten(QMix):
    """Overview:
        QMIX with the mixer replaced by the Qatten mixer; the agent network (DRQN) is unchanged.
    """

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: Union[int, List[int]],
            action_shape: int,
            hidden_size_list: list,
            agent_own_state_size: int,
            mixer: bool = True,
            lstm_type: str = 'gru',
            activation: nn.Module = nn.ReLU(),
            dueling: bool = False,
            n_attention_head: int = 4,
            weighted_head: bool = True,
    ) -> None:
        super(Qatten, self).__init__(
            agent_num, obs_shape, global_obs_shape, action_shape, hidden_size_list,
            mixer=mixer, lstm_type=lstm_type, activation=activation, dueling=dueling
        )
        if self.mixer:
            assert isinstance(global_obs_shape, int), "Qatten only supports a flat global state"
            self._mixer = QattenMixer(
                agent_num, global_obs_shape, agent_own_state_size,
                n_attention_head=n_attention_head, weighted_head=weighted_head
            )
            self._global_state_encoder = nn.Identity()
