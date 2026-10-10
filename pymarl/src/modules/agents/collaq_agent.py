# CollaQ agent, ported from the DI-engine model (ding/model/template/collaq.py).
#   Q = Q_alone(o_alone) + Q_colla(att(o)) - Q_colla_alone(att(o_ally_zeroed))
# Three GRU streams (colla / colla_alone share weights, separate hidden). dueling=False heads,
# 1-head attention inserting [q_out, query-residual] (2*att dims) at the ally/own boundary —
# including the DI-engine detail that the second inserted block is the projected-query residual.
import torch
import torch.nn as nn
import torch.nn.functional as F


class CollaQAttention(nn.Module):
    """DI-engine CollaQMultiHeadAttention with n_head=1, d_k=d_v=d_out=attention_size."""

    def __init__(self, d_model_q, d_model_v, att):
        super().__init__()
        self.att = att
        self.w_qs = nn.Linear(d_model_q, att)
        self.w_ks = nn.Linear(d_model_v, att)
        self.w_vs = nn.Linear(d_model_v, att)
        self.fc1 = nn.Sequential(nn.Linear(att, att), nn.ReLU())
        self.fc2 = nn.Linear(att, att)
        self.layer_norm_q = nn.LayerNorm(att, eps=1e-6)
        self.layer_norm_k = nn.LayerNorm(att, eps=1e-6)
        self.layer_norm_v = nn.LayerNorm(att, eps=1e-6)

    def forward(self, q, k, v):
        B, len_q, len_k = q.size(0), q.size(1), k.size(1)
        q = self.w_qs(q)                      # (B, Lq, att)
        k = self.w_ks(k)
        v = self.w_vs(v)
        residual = q
        qn = self.layer_norm_q(q)
        kn = self.layer_norm_k(k)
        vn = self.layer_norm_v(v)
        attn = torch.matmul(qn, kn.transpose(-2, -1)) / (self.att ** 0.5)
        out = torch.matmul(torch.softmax(attn, dim=-1), vn)   # (B, Lq, att)
        out = self.fc2(self.fc1(out))
        return out, residual


class CollaQAgent(nn.Module):

    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        H = args.rnn_hidden_dim
        A = args.n_agents
        att = getattr(args, "attention_size", 32)

        comp = getattr(args, "obs_components", None)
        assert comp is not None, "env obs_components missing"
        move = comp["move"]
        n_en, nf_en = comp["enemy"]
        n_al, nf_al = comp["ally"]
        own = comp["own"]
        obs_dim = args.obs_shape
        assert n_al == A - 1
        assert move + n_en * nf_en + n_al * nf_al + own == obs_dim, "obs layout mismatch"
        self.n_al, self.nf_al = n_al, nf_al
        self.ally_start = move + n_en * nf_en
        self.ally_end = self.ally_start + n_al * nf_al
        self.own = own                          # the self range is the own block only

        self.attn = CollaQAttention(d_model_q=own, d_model_v=nf_al, att=att)
        colla_in = input_shape + 2 * att        # obs + inserted [q_out, residual]
        alone_in = input_shape - n_al * nf_al

        def head():                             # DiscreteHead(layer_num=1), dueling=False
            return nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, args.n_actions))

        self.colla_fc = nn.Linear(colla_in, H)
        self.colla_rnn = nn.GRUCell(H, H)
        self.colla_head = head()
        self.alone_fc = nn.Linear(alone_in, H)
        self.alone_rnn = nn.GRUCell(H, H)
        self.alone_head = head()


    def init_hidden(self):
        # (1, 3, H): stream order = [alone, colla, colla_alone]
        return self.colla_fc.weight.new(1, 3, self.args.rnn_hidden_dim).zero_()

    def _att_insert(self, flat):
        """obs' = [x[:ally_end], q_out, residual, x[ally_end:]] (+2*att dims)."""
        ally = flat[:, self.ally_start:self.ally_end].view(-1, self.n_al, self.nf_al)
        own = flat[:, self.ally_end:self.ally_end + self.own].unsqueeze(1)   # (BA,1,own)
        q_out, residual = self.attn(own, ally, ally)
        return torch.cat([flat[:, :self.ally_end], q_out.squeeze(1), residual.squeeze(1),
                          flat[:, self.ally_end:]], dim=-1)

    def forward(self, inputs, hidden_state, return_extras=False):
        b, a, e = inputs.size()
        flat = inputs.reshape(b * a, e)
        h3 = hidden_state.reshape(-1, 3, self.args.rnn_hidden_dim)

        pad = flat.clone()
        pad[:, self.ally_start:self.ally_end] = 0.0
        alone = torch.cat([flat[:, :self.ally_start], flat[:, self.ally_end:]], dim=-1)

        h_alone = self.alone_rnn(F.relu(self.alone_fc(alone), inplace=True), h3[:, 0])
        q_alone = self.alone_head(h_alone)
        colla_in = self._att_insert(flat)
        colla_alone_in = self._att_insert(pad)
        h_colla = self.colla_rnn(F.relu(self.colla_fc(colla_in), inplace=True), h3[:, 1])
        q_colla = self.colla_head(h_colla)
        h_ca = self.colla_rnn(F.relu(self.colla_fc(colla_alone_in), inplace=True), h3[:, 2])
        q_ca = self.colla_head(h_ca)

        q = q_alone + q_colla - q_ca
        h_out = torch.stack([h_alone, h_colla, h_ca], dim=1).view(b, a, 3, -1)
        if return_extras:
            return q.view(b, a, -1), q_ca.view(b, a, -1), h_out
        return q.view(b, a, -1), h_out
