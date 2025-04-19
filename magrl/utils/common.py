import numpy as np
import random
import os
import importlib

algorithm_paths = {
    "idqn": ("algorithms.Trainer.idqn", "IDQNAgentTrainer"),
    "irainbow": ("algorithms.Trainer.irainbow", "RainbowAgentTrainer"),
    "maac": ("algorithms.Trainer.maac", "MAACAgentTrainer"),
    "maddpg": ("algorithms.Trainer.maddpg", "MADDPGAgentTrainer"),
    "masac": ("algorithms.Trainer.masac", "MASACAgentTrainer"),
    "mappo": ("algorithms.Trainer.mappo", "MAPPOAgentTrainer"),
    "masac_discrete": ("algorithms.Trainer.masac_discrete", "MASACDiscreteAgentTrainer"),
    "matd3": ("algorithms.Trainer.matd3", "MATD3DiscreteAgentTrainer"),
    "qmix": ("algorithms.Trainer.qmix", "QmixAgentTrainer"),
}

config_paths = {
    "idqn": ("algorithms.Config.Config", "idqn_config"),
    "irainbow": ("algorithms.Config.Config", "irainbow_config"),
    "maac": ("algorithms.Config.Config", "maac_config"),
    "maddpg": ("algorithms.Config.Config", "maddpg_config"),
    "masac": ("algorithms.Config.Config", "masac_config"),
    "mappo": ("algorithms.Config.Config", "mappo_config"),
    "masac_discrete": ("algorithms.Config.Config", "masac_discrete_config"),
    "matd3": ("algorithms.Config.Config", "matd3_config"),
    "qmix": ("algorithms.Config.Config", "qmix_config"),
}
env_prepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "envs")
envs_paths = {"uav_5v5": os.path.join(env_prepath, r"custom_env\uav_5v5\uav_5v5.py"),
              }


# seed
def set_global_random_seed(seed):
    np.random.seed(seed)

    random.seed(seed)
    try:
        import tensorflow as tf
        tf.set_random_seed(seed)
    except Exception as e:
        print("Warning:", e, "no have tensorflow, no need set tf.random.seed")

    try:
        import torch
        torch.manual_seed(seed)
    except Exception as e:
        print("Warning:", e, "no have torch, no need set torch.manual_seed")


# TODO: 未适配动态加载 algorithm class
def get_rl_algorithm(algorithm_name):
    algorithm_path, algorithm_class = algorithm_paths.get(algorithm_name, None)
    config_path, config_class = config_paths.get(algorithm_name, None)
    if algorithm_path and config_path:
        moduleTrainer = importlib.import_module(algorithm_path)
        trainer = getattr(moduleTrainer, algorithm_class)

        moduleCfg = importlib.import_module(config_path)
        cfg = getattr(moduleCfg, config_class)

    else:
        trainer, cfg = None, None
    return trainer, cfg
