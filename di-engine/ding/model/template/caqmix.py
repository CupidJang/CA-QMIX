from typing import Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from ding.utils import list_split, MODEL_REGISTRY
from ding.torch_utils import fc_block, MLP, ScaledDotProductAttention
from ..common.head import DuelingHead
from .q_learning import DRQN
from .qmix import Mixer as qmix_mixer
        
# Multi-head attention with layer-normalised inputs
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_head: int,
        d_model_q: int,
        d_model_v: int,
        d_k: int,
        d_v: int,
        d_out: int,
        dropout: float = 0.,
        activation: nn.Module = nn.ReLU()
    ):
        super(MultiHeadAttention, self).__init__()

        self.act = activation

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model_q, n_head * d_k)
        self.w_ks = nn.Linear(d_model_v, n_head * d_k)
        self.w_vs = nn.Linear(d_model_v, n_head * d_v)

        self.fc1 = fc_block(n_head * d_v, n_head * d_v, activation=self.act)
        self.fc2 = fc_block(n_head * d_v, d_out)

        self.attention = ScaledDotProductAttention(d_k=d_k)
        self.layer_norm_q_in  = nn.LayerNorm(d_model_q)
        self.layer_norm_kv_in = nn.LayerNorm(d_model_v)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        q = self.layer_norm_q_in(q)
        if k is v:
            # Callers pass the same tensor as keys and values; normalise it once.
            k = v = self.layer_norm_kv_in(k)
        else:
            k = self.layer_norm_kv_in(k)
            v = self.layer_norm_kv_in(v)

        q = self.w_qs(q).view(batch_size, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, len_k, n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(batch_size, len_v, n_head, d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)
        out = self.attention(q, k, v, mask=mask)

        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        return self.fc2(self.fc1(out))

# Attention from each agent (query) to its visible allies and itself (keys/values)
class CAQMIXSMACAttentionModule(nn.Module):
    def __init__(self, q_dim, agent_num, self_feature_range, ally_feature_range,
                 attention_size, activation=nn.ReLU()):
        super().__init__()
        self.self_feature_range = self_feature_range
        self.ally_feature_range = ally_feature_range

        ally_span = ally_feature_range[1] - ally_feature_range[0]
        assert ally_span % (agent_num - 1) == 0, (
            f"ally_feature_range {ally_feature_range} (length {ally_span}) is not divisible by "
            f"the number of allies ({agent_num - 1}); check the config."
        )
        ally_dim_per = ally_span // (agent_num - 1)
        self_dim     = self_feature_range[1] - self_feature_range[0]
        self.ally_dim_per = ally_dim_per

        self.ally_proj = nn.Linear(ally_dim_per, attention_size)
        self.self_proj = nn.Linear(self_dim,     attention_size)

        # Keys and values are all projected to attention_size
        self.attention_layer = MultiHeadAttention(
            n_head=2,
            d_model_q=q_dim + self_dim,          
            d_model_v=attention_size,
            d_k=attention_size // 2,
            d_v=attention_size // 2,
            d_out=attention_size,
            activation=activation,
        )

    def _cut_obs(self, obs):
        s = obs[..., self.self_feature_range[0]:self.self_feature_range[1]]
        a = obs[..., self.ally_feature_range[0]:self.ally_feature_range[1]]
        return s, a

    def forward(self, input, hidden, mask=None):
        self_feature, ally_features = self._cut_obs(input)
        T, B, A, _ = hidden.shape

        # Keys/values: projected ally tokens plus a projected self token
        ally_features = ally_features.reshape(T, B, A, A - 1, self.ally_dim_per)

        ally_tokens = self.ally_proj(ally_features)
        self_kv_token = self.self_proj(self_feature).unsqueeze(3)

        kv_seq = torch.cat([ally_tokens, self_kv_token], dim=3)
        kv_seq = kv_seq.reshape(T * B * A, A, -1)

        query = torch.cat([hidden, self_feature], dim=-1)
        query = query.reshape(T * B * A, 1, -1)

        if mask is not None:
            mask = mask.reshape(T * B * A, 1, A)

        out = self.attention_layer(query, kv_seq, kv_seq, mask)
        return out.reshape(T, B, A, -1)
    
@MODEL_REGISTRY.register('caqmix')
class CAQMix(nn.Module):

    def __init__(
            self,
            agent_num: int,
            obs_shape: int,
            global_obs_shape: Union[int, List[int]],
            self_feature_range: Union[int, List[int]],
            ally_feature_range: Union[int, List[int]],
            action_shape: int,
            hidden_size_list: list,
            attention_size : int,
            mixer: bool = True,
            alpha: bool = True,
            coporate: bool = True,
            lstm_type: str = 'gru',
            activation: nn.Module = nn.ReLU(),
            dueling: bool = False,
            gate_variant: str = 'full',
            center_coop: bool = False,
    ) -> None:

        super(CAQMix, self).__init__()
        embedding_size = hidden_size_list[-1]
        self.ally_feature_range = ally_feature_range
        self.alpha = alpha
        self.coporate = coporate
        self.mixer = mixer
        assert gate_variant in ('full', 'no_hard_mask', 'binary', 'no_vis_input', 'shuffled_vis'), \
            f"unknown gate_variant: {gate_variant}"
        self.gate_variant = gate_variant
        # center_coop: subtract the action mean from Q^coop. Its state-value part cannot change the greedy action
        # and only inflates agent_q, so centring leaves Q^coop as a pure preference adjustment. Default: off.
        self.center_coop = center_coop

        if self.alpha and not self.coporate:
            raise ValueError("alpha=True requires coporate=True (uses attention_hidden_output)")
        
        # agent_q network
        self._q_network = DRQN(
            obs_shape, action_shape, hidden_size_list, lstm_type=lstm_type, dueling=dueling, activation=activation
        )

        # Cooperative Q-value: attention context + agent hidden state -> dueling head (V_coop + advantage)
        if self.coporate:
            self._self_attention = CAQMIXSMACAttentionModule(
                q_dim=embedding_size,
                agent_num=agent_num,
                self_feature_range=self_feature_range,
                ally_feature_range=ally_feature_range,
                attention_size=attention_size,
                activation=activation,
            )

            self.layernorm = nn.LayerNorm(attention_size + embedding_size)
            self.residual_layer = nn.Sequential(
                nn.Linear(attention_size + embedding_size, 2 * (attention_size + embedding_size)),
                activation,
                nn.Linear(2 * (attention_size + embedding_size),  attention_size + embedding_size)
            )

            self.dueling_corporate_q = DuelingHead(hidden_size= attention_size + embedding_size , output_size=action_shape)
    
        # alpha network
        if self.alpha:
            self.alpha_network = nn.Sequential(
                nn.Linear(embedding_size + (agent_num - 1) + attention_size, embedding_size),
                activation,
                nn.Linear(embedding_size, 1)
            )
        # mixer
        if self.mixer:
            global_obs_shape_type = self._get_global_obs_shape_type(global_obs_shape)
            
            if global_obs_shape_type == "flat":
                self._mixer = qmix_mixer(agent_num, global_obs_shape, embedding_size, activation=activation)
                self._global_state_encoder = nn.Identity()
            else:
                raise ValueError(f"Unsupported global_obs_shape: {global_obs_shape}")
    
    def _get_global_obs_shape_type(self, global_obs_shape: Union[int, List[int]]) -> str:
        if isinstance(global_obs_shape, int) or (isinstance(global_obs_shape, list) and len(global_obs_shape) == 1):
            return "flat"
        else:
            raise ValueError(f"Unsupported global_obs_shape: {global_obs_shape}")
    
        
    def forward(self, data: dict, single_step: bool = True,
                need_mixer: bool = True, need_delta_q: bool = True):
        """
        need_mixer=False:   only the logits are needed (action selection for double Q). Skips the mixer, the global
                            state encoding and the max path used by the Delta-Q term.
        need_delta_q=False: only total_q is needed (target network). Skips the two extra mixer passes for Delta-Q.
        Both flags only drop outputs that are not used, so losses and gradients are unchanged.
        """
        agent_state, global_state, prev_state = data['obs']['agent_state'], data['obs']['global_state'], data[
            'prev_state']
        action = data.get('action', None)
        if single_step:
            agent_state = agent_state.unsqueeze(0)
        if single_step and len(global_state.shape) == 2:
            global_state = global_state.unsqueeze(0)
        T, B, A = agent_state.shape[:3]

        assert len(prev_state) == B and all(
            [len(p) == A for p in prev_state]
        ), '{}-{}-{}-{}'.format([type(p) for p in prev_state], B, A, len(prev_state[0]))
        
        prev_state = reduce(lambda x, y: x + y, prev_state)

        agent_state = agent_state.reshape(T, -1, *agent_state.shape[3:])
        output = self._q_network({'obs': agent_state, 'prev_state': prev_state, 'enable_fast_timestep': True})
        before_agent_q, next_state , hidden = output['logit'], output['next_state'] , output["hidden_state"]
        next_state, _ = list_split(next_state, step=A)

        before_agent_q = before_agent_q.reshape(T, B, A, -1) # (T , B , A , P)
        agent_state_input = agent_state.reshape(T , B , A , -1)
        hidden = hidden.reshape(T , B , A , -1) # (T , B , A , E)

        action_mask = data['obs']['action_mask']
        if len(action_mask.shape) == 3:
            action_mask = action_mask.unsqueeze(0)        # (T, B, A, action_shape)

        ally_mask = None
        has_visible = None
        if self.coporate and self.ally_feature_range is not None:
            start, end = self.ally_feature_range
            ally_dim = end - start
            feats_per_ally = ally_dim // (A - 1)
            ally_slice = agent_state[..., start:end]
            ally_reshaped = ally_slice.reshape(T, B, A, A - 1, feats_per_ally)
            ally_mask = (ally_reshaped[..., 0:1] > 0.5).float()         # (T,B,A,A-1,1)
            has_visible = (ally_mask.sum(dim=3) > 0).float()            # (T,B,A,1)

        attention_context_output = None
        coop_value = None
        if self.coporate:
            ally_visible    = ally_mask.squeeze(-1).bool()                    # (T,B,A,A-1)
            self_mask = (has_visible == 0).bool()  
            full_mask = torch.cat([ally_visible, self_mask], dim=-1)

            attention_context_output = self._self_attention(agent_state_input, hidden, mask=full_mask)
            combine = torch.cat([attention_context_output , hidden] , dim = -1)
            corporate_input = combine + self.residual_layer(self.layernorm(combine))
            coop_value = self.dueling_corporate_q(corporate_input)['logit']
            if self.center_coop:
                coop_value = coop_value - coop_value.mean(dim=-1, keepdim=True)

        alpha = None
        if self.alpha:
            mask_float = ally_mask.squeeze(-1).float()          # (T,B,A,A-1)
            gate_has_visible = has_visible                      # used for the hard mask

            if self.gate_variant == 'shuffled_vis':
                # Shuffle visibility across the batch: keeps how often allies are visible but breaks its
                # alignment with the current context. The attention mask is left unchanged.
                perm = torch.randperm(mask_float.shape[1], device=mask_float.device)
                mask_float = mask_float[:, perm]
                gate_has_visible = (mask_float.sum(dim=-1, keepdim=True) > 0).float()
            elif self.gate_variant == 'no_vis_input':
                # Remove the visibility input m from the gate MLP (zero-filled to keep the parameter count).
                mask_float = torch.zeros_like(mask_float)

            if self.gate_variant == 'binary':
                # Gate from visibility alone, without the sigmoid MLP
                alpha = gate_has_visible
            else:
                alpha_input = torch.cat([hidden, mask_float, attention_context_output], dim=-1)
                alpha = torch.sigmoid(self.alpha_network(alpha_input))
                if self.gate_variant != 'no_hard_mask':
                    alpha = alpha * gate_has_visible            # (T,B,A,1)

        agent_q = before_agent_q
        if self.coporate:
            if self.alpha:
                agent_q = agent_q + alpha * coop_value
            else:
                agent_q = agent_q + coop_value 

        if action is None:
            # Greedy action under the action mask (no action given)
            agent_q = agent_q.masked_fill(action_mask == 0.0, -1e9)
            action = agent_q.argmax(dim=-1)

        agent_q_act = torch.gather(agent_q, dim=-1, index=action.unsqueeze(-1))
        agent_q_act = agent_q_act.squeeze(-1)  # (T, B, A): Q-value of the chosen action

        total_q = None
        total_q_base_max, total_q_final_max = None, None
        masked_before_q = masked_agent_q = None
        # The Delta-Q max path is only built for sequence calls that need it
        want_dq = (not single_step) and need_delta_q

        if not need_mixer:
            # Only the logits are needed: skip the mixer path entirely
            pass
        elif self.mixer:
            global_state_embedding = self._process_global_state(global_state)
            total_q = self._mixer(agent_q_act, global_state_embedding)

            # Inputs of the Delta-Q loss (monotonic improvement constraint)
            if want_dq:
                masked_before_q = before_agent_q.masked_fill(action_mask == 0.0, -1e9)
                masked_agent_q = agent_q.masked_fill(action_mask == 0.0, -1e9)
                total_q_base_max = self._mixer(masked_before_q.max(dim=-1)[0], global_state_embedding)
                total_q_final_max = self._mixer(masked_agent_q.max(dim=-1)[0], global_state_embedding)
        else:
            # Without a mixer, sum the agent Q-values (VDN), as in the QMIX model.
            total_q = agent_q_act.sum(dim=-1)
            if want_dq:
                masked_before_q = before_agent_q.masked_fill(action_mask == 0.0, -1e9)
                masked_agent_q = agent_q.masked_fill(action_mask == 0.0, -1e9)
                total_q_base_max = masked_before_q.max(dim=-1)[0].sum(dim=-1)
                total_q_final_max = masked_agent_q.max(dim=-1)[0].sum(dim=-1)

        if single_step:
            agent_q = agent_q.squeeze(0)
            if total_q is not None:
                total_q = total_q.squeeze(0)
        
        output_dict = {
            'total_q': total_q,
            'logit': agent_q,
            'next_state': next_state,
            'action_mask': data['obs']['action_mask']
        }

        if not single_step:
            output_dict['before_q'] = before_agent_q
            output_dict['total_q_base_max'] = total_q_base_max
            output_dict['total_q_final_max'] = total_q_final_max
            # Leave these keys out when the Delta-Q path was skipped: the policy's monitoring code checks
            # for their presence and takes an argmax over them.
            if masked_before_q is not None:
                output_dict['masked_before_q'] = masked_before_q
                output_dict['masked_final_q'] = masked_agent_q

            if self.coporate:
                output_dict['coop_value']  = coop_value         # (T,B,A,action_shape)
                output_dict['has_visible'] = has_visible
                output_dict['ally_mask']   = ally_mask

            if self.alpha:
                output_dict['alpha'] = alpha

        return output_dict

    def _process_global_state(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        Overview:
            Process the global state to obtain an embedding.
        Arguments:
            - global_state (:obj:`torch.Tensor`): The global state tensor.

        Returns:
            - global_state_embedding (:obj:`torch.Tensor`): The processed global state embedding.
        """
        # If global_state has 5 dimensions, it's likely in the form [batch_size, time_steps, C, H, W]
        if global_state.dim() == 5:
            # Reshape and apply the global state encoder
            batch_time_shape = global_state.shape[:2]  # [batch_size, time_steps]
            reshaped_state = global_state.view(-1, *global_state.shape[-3:])  # Collapse batch and time dims
            encoded_state = self._global_state_encoder(reshaped_state)
            return encoded_state.view(*batch_time_shape, -1)  # Reshape back to [batch_size, time_steps, embedding_dim]
        else:
            # For lower-dimensional states, apply the encoder directly
            return self._global_state_encoder(global_state)