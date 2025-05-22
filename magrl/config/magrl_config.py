#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## 配置文件
import os
import torch

from . import Singleton

# 定义训练所需的参数：
#     环境相关参数
#     训练用的超参数定义
#     checkpointing（用于存储数据和模型）
#     测试阶段的参数
# Environment环境相关参数


class TrainConfig(metaclass=Singleton):
    def __init__(self):
        # 用tf1还是torch， 默认torch
        self.nn_package = "torch"

        self.scenario_name = "uav_5v5"  # 场景
        self.num_adversaries = 5  # red
        self.num_agents = 5  # blue

        self.total_num_agent = self.num_adversaries + self.num_agents
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 算法配置参数
        self.algo_name = 'maddpg'
        self.network_structure = 'share' if 'share' in self.algo_name else 'no_share'
        self.update_step = 10
        self.learn_type = 'off-policy'
        self.algo_config = {'red': None, 'blue': None}
        self.adv_policy = "CTDE"
        self.good_policy = "CTDE"
        self.adv_algorithm = "masac"
        self.good_algorithm = "masac"

        self.seed = 251  # 幸运种子
        # 回合数
        self.epochs = 1
        self.episodes = 20000
        self.max_episode_len = 300  #
        # self.td_lammda = 0.8
        # self.lr = 1e-4
        # self.gamma = 0.99  # 56, 114, 229, 459 的max_episode_len 对应 0.96, 0.98, 0.99, 0.995 的gamma

        # 保存配置参数
        self.load_model = False
        self.save_name = f"{self.num_adversaries}v{self.num_agents}_{self.scenario_name}"
        main_prepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.save_model_dir = os.path.join(main_prepath, r"metrics/model/save")
        self.save_data_dir = os.path.join(main_prepath, r"metrics/data/learning_curves")
        self.save_rate = 2
        self.load_dir = os.path.join(main_prepath, r"metrics/model/load")

        self.data_file_dir = 'data_name.pkl'
        self.create_xslx_dir = 'data_name.xlsx'
        self.plot_dir = "./metrics/"

        # 特殊算法固有的参数
        self.init_alpha = 0.02
        self.fix_alpha = True
        self.use_target_actor = True
        self.epsilon = 1e-6


def get_config(env_name, algo_name):
    r"""

    根据env_name, algo_name提取对应的config文件，返回config超参数

    :param env_name:
    :param algo_name:
    :return:
    """
    config = {}
    return config

