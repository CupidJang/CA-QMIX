import copy
import inspect
from collections import namedtuple
from typing import Dict, List

import numpy as np
from easydict import EasyDict
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from ding.envs import BaseEnv
from ding.envs.common.env_element import EnvElementInfo
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY, deep_merge_dicts

# Unit-type and start-position distributions of the SMACv2 10-vs-10 scenarios.
SMAC2_PRESET_CONFIGS = {
    "10gen_terran": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["marine", "marauder", "medivac"],
            "exception_unit_types": ["medivac"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 10,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "10gen_zerg": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["zergling", "baneling", "hydralisk"],
            "exception_unit_types": ["baneling"],
            "weights": [0.45, 0.1, 0.45],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 10,
            "map_x": 32,
            "map_y": 32,
        },
    },
    "10gen_protoss": {
        "n_units": 10,
        "n_enemies": 10,
        "team_gen": {
            "dist_type": "weighted_teams",
            "unit_types": ["stalker", "zealot", "colossus"],
            "exception_unit_types": ["colossus"],
            "weights": [0.45, 0.45, 0.1],
            "observe": True,
        },
        "start_positions": {
            "dist_type": "surrounded_and_reflect",
            "p": 0.5,
            "n_enemies": 10,
            "map_x": 32,
            "map_y": 32,
        },
    },
}


@ENV_REGISTRY.register("smac2")
class SMAC2Env(BaseEnv):
    """
    DI-engine wrapper of the SMACv2 ``StarCraftCapabilityEnvWrapper``.
    Observations are dicts with ``agent_state`` (n_agents, obs_dim), ``global_state`` (state_dim,)
    and ``action_mask`` (n_agents, n_actions). With ``obs_alone=True`` the ally block is also
    removed (``agent_alone_state``) or zeroed (``agent_alone_padding_state``) for CollaQ.
    """
    config = dict(
        map_name=None,
        difficulty="7",
        seed=0,
        obs_alone=False,
        # reward (same arguments as StarCraft2Env)
        reward_only_positive=True,
        reward_sparse=False,
        reward_death_value=10,
        reward_win=200,
        reward_defeat=0,
        reward_negative_scale=0.5,
        reward_scale=True,
        reward_scale_rate=20,
        # observation
        step_mul=8,
        obs_last_action=True,
        obs_all_health=True,
        obs_own_health=True,
        obs_pathing_grid=False,
        obs_terrain_height=False,
        obs_timestep_number=False,
        obs_own_pos=False,
        obs_starcraft=True,
        state_last_action=True,
        state_timestep_number=False,
        # StarCraft II
        debug=False,
        window_size_x=192,
        window_size_y=120,
        replay_dir="",
        replay_prefix="",
        heuristic_ai=False,
        fully_observable=False,
        continuing_episode=False,
        use_unit_ranges=False,
        min_attack_range=2,
        conic_fov=False,
        num_fov_actions=12,
        prob_obs_enemy=1.0,
    )

    SMAC2Timestep = namedtuple('SMAC2Timestep', ['obs', 'reward', 'done', 'info'])
    SMAC2EnvInfo = namedtuple('SMAC2EnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space'])

    def __init__(self, cfg: dict) -> None:
        cfg = deep_merge_dicts(EasyDict(self.config), cfg)
        self.cfg = cfg
        assert cfg.map_name in SMAC2_PRESET_CONFIGS, f"unknown SMACv2 map: {cfg.map_name}"
        self.map_name = cfg.map_name
        self.difficulty = cfg.difficulty
        self._capability_config = copy.deepcopy(SMAC2_PRESET_CONFIGS[self.map_name])
        self.n_agents = self._capability_config["n_units"]
        self.n_enemies = self._capability_config["n_enemies"]

        self._env = self._make_env()
        self._init_flag = True

        self.obs_alone = cfg.obs_alone
        if self.obs_alone:
            self._setup_obs_alone()

        self.reward_death_value = cfg.reward_death_value
        self.reward_win = cfg.reward_win
        self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win
        self.reward_scale_rate = cfg.reward_scale_rate

        self._observation_space = self.get_obs_space()
        self._reward_space = self.reward_info()
        self._action_space = self.action_info()

    def _make_env(self) -> StarCraftCapabilityEnvWrapper:
        return StarCraftCapabilityEnvWrapper(
            capability_config=self._capability_config,
            map_name=self.map_name,
            difficulty=self.difficulty,
            **self._sc2_kwargs(),
        )

    def _sc2_kwargs(self) -> dict:
        """The subset of the config accepted by smacv2's StarCraft2Env."""
        accepted = set(inspect.signature(StarCraft2Env.__init__).parameters) - {'self', 'kwargs'}
        skip = {'map_name', 'capability_config', 'difficulty', 'seed'}
        return {k: self.cfg[k] for k in accepted - skip if k in self.cfg}

    def reset(self) -> Dict[str, np.ndarray]:
        if not self._init_flag:
            self._env = self._make_env()
            self._init_flag = True
        # On a failed unit spawn the smacv2 wrapper retries internally but drops the
        # return value, so the observation is read back from the environment instead.
        ret = self._env.reset()
        obs, state = ret if ret is not None else (self._env.get_obs(), self._env.get_state())
        return self._apply_obs_alone({
            "agent_state": np.array(obs),
            "global_state": np.array(state),
            "action_mask": np.array(self._env.get_avail_actions()),
        })

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        reward, terminated, info = self._env.step(actions)
        if terminated:
            info["eval_episode_return"] = 1.0 if info.get("battle_won", False) else 0.0
        obs_dict = self._get_obs_dict()
        reward = to_ndarray([reward], dtype=np.float32)
        return self.SMAC2Timestep(obs=obs_dict, reward=reward, done=terminated, info=info)

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def seed(self, seed: int, dynamic_seed: bool = False) -> None:
        self._seed = seed

    def get_obs_space(self):
        obs_dict_shape = {
            'agent_state': (self.n_agents, self._env.get_obs_size()),
            'global_state': (self._env.get_state_size(), ),
            'action_mask': (self.n_agents, self._env.get_total_actions()),
        }
        if self.obs_alone:
            obs_dict_shape['agent_alone_state'] = (self.n_agents, self._alone_obs_size)
            obs_dict_shape['agent_alone_padding_state'] = (self.n_agents, self._env.get_obs_size())
        return EnvElementInfo(obs_dict_shape, None)

    def action_info(self):
        return (self._env.get_total_actions(), ), {'min': 0, 'max': 1}

    def reward_info(self):
        return (1, ), {'min': 0, 'max': self.max_reward / self.reward_scale_rate}, None, None

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        return self._apply_obs_alone({
            "agent_state": np.array(self._env.get_obs()),
            "global_state": np.array(self._env.get_state()),
            "action_mask": np.array(self._env.get_avail_actions()),
        })

    def _setup_obs_alone(self) -> None:
        """Locates the ally block in the per-agent observation [move | enemy | ally | own]."""
        move = self._env.get_obs_move_feats_size()
        n_en, nf_en = self._env.get_obs_enemy_feats_size()
        n_al, nf_al = self._env.get_obs_ally_feats_size()
        nf_own = self._env.get_obs_own_feats_size()
        self._ally_start = move + n_en * nf_en
        self._ally_end = self._ally_start + n_al * nf_al
        self._alone_obs_size = self._env.get_obs_size() - (self._ally_end - self._ally_start)
        self.ally_feature_range = [self._ally_start, self._ally_end]
        self.self_feature_range = [self._ally_end, self._ally_end + nf_own]

    def _apply_obs_alone(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Adds the ally-free observations used by CollaQ (no-op if obs_alone=False)."""
        if not self.obs_alone:
            return obs_dict
        obs = np.asarray(obs_dict["agent_state"], dtype=np.float32)
        s, e = self._ally_start, self._ally_end
        padding = obs.copy()
        padding[:, s:e] = 0.0
        obs_dict["agent_state"] = obs
        obs_dict["agent_alone_state"] = np.concatenate([obs[:, :s], obs[:, e:]], axis=1)
        obs_dict["agent_alone_padding_state"] = padding
        return obs_dict

    @staticmethod
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        return [copy.deepcopy(cfg) for _ in range(cfg.get("collector_env_num", 1))]

    @staticmethod
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        evaluator_cfg = copy.deepcopy(cfg)
        evaluator_cfg["seed"] = cfg.get("seed", 0)
        return [evaluator_cfg for _ in range(cfg.get("evaluator_env_num", 1))]

    def __repr__(self):
        return "DI-engine SMACv2 Env"
