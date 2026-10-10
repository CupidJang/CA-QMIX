# Modified for CA-QMIX: yaml FullLoader, collections.abc, single run entry point, paper settings (config/paper.yaml).
import numpy as np
import os
import collections
from os.path import dirname, abspath, join
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import REGISTRY as run_REGISTRY

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]
    
    # run
    run_REGISTRY[_config['run']](_run, config, _log)

def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


def parse_command(params, key, default):
    result = default
    for _i, _v in enumerate(params):
        if _v.split("=")[0].strip() == key:
            result = _v[_v.index('=')+1:].strip()
            break
    return result


def apply_paper_settings(config_dict, params):
    """Applies config/paper.yaml: the per-map settings, then the algorithm's per-map or ablation settings."""
    with open(os.path.join(os.path.dirname(__file__), "config", "paper.yaml"), "r") as f:
        paper = yaml.load(f, Loader=yaml.FullLoader)
    map_name = parse_command(params, "env_args.map_name", config_dict["env_args"]["map_name"])
    ablation = parse_command(params, "ablation", config_dict.get("ablation"))
    alg = paper.get(config_dict["name"], {})
    updates = [paper["maps"].get(map_name, {})]
    if ablation in (None, "None", "null"):
        updates.append(alg.get("maps", {}).get(map_name, {}))
    else:
        assert ablation in alg.get("ablations", {}), "unknown ablation for {}: {}".format(config_dict["name"], ablation)
        updates.append(alg["ablations"][ablation])
        config_dict["name"] = "{}_{}".format(config_dict["name"], ablation)
    for u in updates:
        config_dict = recursive_dict_update(config_dict, deepcopy(u))
    return config_dict


def build_config(params):
    """default.yaml <- env config <- algorithm config <- paper settings. Removes --config/--env-config from params."""
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)
    return apply_paper_settings(config_dict, params)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    config_dict = build_config(params)

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    map_name = parse_command(params, "env_args.map_name", config_dict['env_args']['map_name'])
    algo_name = parse_command(params, "name", config_dict['name']) 
    file_obs_path = join(results_path, "sacred", map_name, algo_name)
    
    logger.info("Saving to FileStorageObserver in {}.".format(file_obs_path))
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)

    # flush
    sys.stdout.flush()
