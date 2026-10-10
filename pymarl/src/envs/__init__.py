# Modified for CA-QMIX: StarCraft II only; the env info exposes the observation block sizes.
from functools import partial
import sys
import os

from .multiagentenv import MultiAgentEnv
from .starcraft import StarCraft2Env


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


class SC2EnvWithComponents(StarCraft2Env):
    """Exposes the observation block sizes so agents can slice ally and own features."""

    def get_env_info(self):
        info = super().get_env_info()
        info["obs_components"] = {
            "move": self.get_obs_move_feats_size(),
            "enemy": self.get_obs_enemy_feats_size(),
            "ally": self.get_obs_ally_feats_size(),
            "own": self.get_obs_own_feats_size(),
        }
        return info


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=SC2EnvWithComponents)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")
