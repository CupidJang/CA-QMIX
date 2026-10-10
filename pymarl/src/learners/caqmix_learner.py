# CA-QMIX learner: original-pymarl q_learner semantics (1-step TD, double-Q, hard target update)
# plus the Delta-Q regulariser of the DI-engine CA-QMIX policy:
#   L_delta = mean over valid t of relu( Qtot(max_a Q_base) - Qtot(max_a Q_adj) ), online mixer.
import copy
import torch as th
from torch.optim import RMSprop, Adam
from components.episode_buffer import EpisodeBatch
from modules.mixers.qmix import QMixer
from modules.mixers.vdn import VDNMixer


class CaqMixLearner:

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters())
        self.last_target_update_episode = 0
        self.lambda_delta_q = getattr(args, "lambda_delta_q", 1.0)

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        if getattr(args, "optimizer", "rmsprop") == "adam":
            self.optimiser = Adam(params=self.params, lr=args.lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr,
                                     alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # online rollout — adjusted and base Q for every timestep
        mac_out, base_out = [], []
        _al_ts, _gt_ts = [], []   # per-step gate values, for logging only
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            q_adj, q_base = self.mac.forward(batch, t=t, return_base=True)
            mac_out.append(q_adj)
            base_out.append(q_base)
            _la = getattr(self.mac.agent, "last_alpha_t", None)
            if _la is not None:
                _al_ts.append(_la.view(batch.batch_size, -1))
                _gt_ts.append(self.mac.agent.last_gate_t.view(batch.batch_size, -1))
        mac_out = th.stack(mac_out, dim=1)      # (B, T, A, n_actions)
        base_out = th.stack(base_out, dim=1)

        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # target rollout (adjusted Q)
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_mac_out.append(self.target_mac.forward(batch, t=t))
        target_mac_out = th.stack(target_mac_out[1:], dim=1)
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        if self.args.double_q:
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        td_error = chosen_action_qvals - targets.detach()
        mask_td = mask.expand_as(td_error)
        masked_td_error = td_error * mask_td
        td_loss = (masked_td_error ** 2).sum() / mask_td.sum()

        # Delta-Q regulariser (online mixer, greedy max over available actions)
        avail_cur = avail_actions[:, :-1]
        if self.lambda_delta_q == 0:
            delta_q = th.zeros_like(mask)
            delta_q_loss = th.zeros((), device=mask.device)
        else:
            base_star = base_out[:, :-1].masked_fill(avail_cur == 0, -1e9).max(dim=3)[0]
            adj_star = mac_out[:, :-1].masked_fill(avail_cur == 0, -1e9).max(dim=3)[0]
            if self.mixer is not None:
                base_star_tot = self.mixer(base_star, batch["state"][:, :-1])
                adj_star_tot = self.mixer(adj_star, batch["state"][:, :-1])
            else:
                base_star_tot = base_star.sum(dim=-1, keepdim=True)
                adj_star_tot = adj_star.sum(dim=-1, keepdim=True)
            delta_q = th.relu(base_star_tot - adj_star_tot)
            delta_q_loss = (delta_q * mask).sum() / mask.sum()

        loss = td_loss + self.lambda_delta_q * delta_q_loss

        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("delta_q_loss", delta_q_loss.item(), t_env)
            self.logger.log_stat("delta_q_pos_ratio", float((delta_q > 0).float().mean().item()), t_env)
            gn = grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm)
            self.logger.log_stat("grad_norm", gn, t_env)
            mask_elems = mask_td.sum().item()
            self.logger.log_stat("td_error_abs", masked_td_error.abs().sum().item() / mask_elems, t_env)
            self.logger.log_stat("alpha_mean", getattr(self.mac.agent, "last_alpha_mean", 0.0), t_env)
            self.logger.log_stat("gate_open_ratio", getattr(self.mac.agent, "last_gate_open", 0.0), t_env)
            try:  # gate statistics over filled timesteps (logging only)
                if _al_ts:
                    _A = th.stack(_al_ts, dim=1)                      # (B,T,A)
                    _G = th.stack(_gt_ts, dim=1)
                    _F = batch["filled"][:, :_A.size(1)].float()      # (B,T,1)
                    _den = (_F.expand_as(_A)).sum().item()
                    _gs = (_G * _F).sum().item()
                    if _den > 0:
                        self.logger.log_stat("alpha_mean_real", (_A * _F).sum().item() / _den, t_env)
                        self.logger.log_stat("gate_open_real", _gs / _den, t_env)
                    if _gs > 0:
                        self.logger.log_stat("alpha_when_vis_real", (_A * _F).sum().item() / _gs, t_env)
            except Exception:
                pass
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda s, l: s))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda s, l: s))
