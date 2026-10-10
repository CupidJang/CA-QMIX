# CA-QMIX agent, ported from the DI-engine model (di-engine/ding/model/template/caqmix.py).
# Base DRQN stream + attention over ally tokens (Q^coop) + visibility-conditioned gate (alpha).
#   Q_adj = Q_base + alpha * Q_coop
# Feature offsets come from env obs_components (see src/envs/__init__.py, src/run/run.py).
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingHead(nn.Module):
    """Same as DI-engine DuelingHead(hidden, out, layer_num=1): q = V + A - mean(A)."""

    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.A = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                               nn.Linear(hidden_size, output_size))
        self.V = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(),
                               nn.Linear(hidden_size, 1))

    def forward(self, x):
        a = self.A(x)
        v = self.V(x)
        return v + a - a.mean(dim=-1, keepdim=True)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)   # mask True = attend
        attn = torch.softmax(attn, dim=-1)
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """Same as the DI-engine CA-QMIX attention: LayerNorm on q/kv inputs, heads, fc1(ReLU) + fc2."""

    def __init__(self, n_head, d_model_q, d_model_v, d_k, d_v, d_out):
        super().__init__()
        self.n_head, self.d_k, self.d_v = n_head, d_k, d_v
        self.w_qs = nn.Linear(d_model_q, n_head * d_k)
        self.w_ks = nn.Linear(d_model_v, n_head * d_k)
        self.w_vs = nn.Linear(d_model_v, n_head * d_v)
        self.fc1 = nn.Sequential(nn.Linear(n_head * d_v, n_head * d_v), nn.ReLU())
        self.fc2 = nn.Linear(n_head * d_v, d_out)
        self.attention = ScaledDotProductAttention(d_k)
        self.layer_norm_q_in = nn.LayerNorm(d_model_q)
        self.layer_norm_kv_in = nn.LayerNorm(d_model_v)

    def forward(self, q, k, v, mask=None):
        B, len_q, len_k = q.size(0), q.size(1), k.size(1)
        q = self.layer_norm_q_in(q)
        if k is v:
            k = v = self.layer_norm_kv_in(k)
        else:
            k = self.layer_norm_kv_in(k)
            v = self.layer_norm_kv_in(v)
        q = self.w_qs(q).view(B, len_q, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_ks(k).view(B, len_k, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_vs(v).view(B, len_k, self.n_head, self.d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)               # broadcast over heads
        out = self.attention(q, k, v, mask=mask)
        out = out.transpose(1, 2).contiguous().view(B, len_q, -1)
        return self.fc2(self.fc1(out))


class CaqMixAgent(nn.Module):

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        H = args.rnn_hidden_dim
        A = args.n_agents
        att = getattr(args, "attention_size", 32)
        self.gate_variant = getattr(args, "gate_variant", "full")
        assert self.gate_variant in ("full", "no_hard_mask", "binary", "no_vis_input", "shuffled_vis")
        # use_alpha=False removes the gate and the hard mask: Q = Q_base + Q_coop
        self.use_alpha = bool(getattr(args, "use_alpha", True))
        self.center_coop = bool(getattr(args, "center_coop", False))

        # feature offsets; fail loudly if the observation layout does not match
        comp = getattr(args, "obs_components", None)
        assert comp is not None, "env obs_components missing — check src/envs/__init__.py plumbing"
        move = comp["move"]
        n_en, nf_en = comp["enemy"]
        n_al, nf_al = comp["ally"]
        own = comp["own"]
        obs_dim = args.obs_shape
        assert n_al == A - 1, f"ally count {n_al} != n_agents-1 {A-1}"
        assert move + n_en * nf_en + n_al * nf_al + own == obs_dim, \
            f"obs layout mismatch: {move}+{n_en}x{nf_en}+{n_al}x{nf_al}+{own} != {obs_dim}"
        self.obs_dim = obs_dim
        self.n_al, self.nf_al = n_al, nf_al
        self.ally_start = move + n_en * nf_en
        self.ally_end = self.ally_start + n_al * nf_al
        # self features = own block + everything the controller appends (last_action, agent_id)
        self.self_dim = input_shape - self.ally_end

        # base DRQN stream
        self.fc1 = nn.Linear(input_shape, H)
        self.rnn = nn.GRUCell(H, H)
        self.base_head = DuelingHead(H, args.n_actions)

        # cooperative branch: attention over allies + residual block + dueling head
        self.ally_proj = nn.Linear(nf_al, att)
        self.self_proj = nn.Linear(self.self_dim, att)
        self.attn = MultiHeadAttention(n_head=2, d_model_q=H + self.self_dim, d_model_v=att,
                                       d_k=att // 2, d_v=att // 2, d_out=att)
        self.layernorm = nn.LayerNorm(att + H)
        self.residual = nn.Sequential(nn.Linear(att + H, 2 * (att + H)), nn.ReLU(),
                                      nn.Linear(2 * (att + H), att + H))
        self.coop_head = DuelingHead(att + H, args.n_actions)

        # gate
        self.alpha_net = nn.Sequential(nn.Linear(H + n_al + att, H), nn.ReLU(), nn.Linear(H, 1))

        self.last_alpha_mean = 0.0
        self.last_gate_open = 0.0
        self.last_alpha_t = None   # (BA, 1), detached; logged by the learner
        self.last_gate_t = None

    def init_hidden(self):
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, return_base=False):
        b, a, e = inputs.size()
        flat = inputs.view(b * a, e)

        x = F.relu(self.fc1(flat), inplace=True)
        h = self.rnn(x, hidden_state.reshape(-1, self.args.rnn_hidden_dim))
        q_base = self.base_head(h)

        obs = flat[:, :self.obs_dim]
        ally = obs[:, self.ally_start:self.ally_end].view(b * a, self.n_al, self.nf_al)
        vis = ally[:, :, 0:1] > 0.5                          # (BA, n_al, 1) bool — visible flag
        has_vis = (vis.sum(dim=1) > 0).float()               # (BA, 1)
        f_self = flat[:, self.ally_end:]                     # own + last_action + agent_id

        ally_tok = self.ally_proj(ally)
        self_tok = self.self_proj(f_self).unsqueeze(1)
        kv = torch.cat([ally_tok, self_tok], dim=1)          # (BA, A, att)
        query = torch.cat([h, f_self], dim=-1).unsqueeze(1)  # (BA, 1, H+self)
        # the self token is attended only when no ally is visible (keeps the softmax valid)
        mask = torch.cat([vis.squeeze(-1), has_vis == 0], dim=1).unsqueeze(1)  # (BA,1,A) True=attend
        ctx = self.attn(query, kv, kv, mask=mask).squeeze(1)  # (BA, att)

        comb = torch.cat([ctx, h], dim=-1)
        corp = comb + self.residual(self.layernorm(comb))
        coop = self.coop_head(corp)
        if self.center_coop:
            coop = coop - coop.mean(dim=-1, keepdim=True)

        vis_f = vis.squeeze(-1).float()
        gate_vis = has_vis
        if self.gate_variant == "shuffled_vis":
            perm = torch.randperm(b, device=flat.device)     # permute across the batch
            vis_f = vis_f.view(b, a, -1)[perm].reshape(b * a, -1)
            gate_vis = (vis_f.sum(dim=-1, keepdim=True) > 0).float()
        elif self.gate_variant == "no_vis_input":
            vis_f = torch.zeros_like(vis_f)

        if not self.use_alpha:
            q = q_base + coop
            self.last_alpha_mean = 1.0
            self.last_gate_open = float(gate_vis.detach().mean())
            self.last_alpha_t = torch.ones_like(gate_vis)
            self.last_gate_t = gate_vis.detach()
        else:
            if self.gate_variant == "binary":
                alpha = gate_vis
            else:
                alpha = torch.sigmoid(self.alpha_net(torch.cat([h, vis_f, ctx], dim=-1)))
                if self.gate_variant != "no_hard_mask":
                    alpha = alpha * gate_vis

            q = q_base + alpha * coop

            self.last_alpha_mean = float(alpha.detach().mean())
            self.last_gate_open = float(gate_vis.detach().mean())
            self.last_alpha_t = alpha.detach()
            self.last_gate_t = gate_vis.detach()

        if return_base:
            return q.view(b, a, -1), q_base.view(b, a, -1), h.view(b, a, -1)
        return q.view(b, a, -1), h.view(b, a, -1)
