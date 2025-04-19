from .. import BaseDRL
import torch
import torch.optim as optim
from ..utils import copy_model_params
import random


class RandomPath(BaseDRL):
    def __init__(self, config, env, idx):
        super(RandomPath, self).__init__(config, env, idx)
        assert self.cur_agent['observation_feature_list'] == [
            "phase_2_num_vehicle",
            "phase_2_num_waiting_vehicle",
            "phase_2_sum_waiting_time",
            "phase_2_delay",
            "phase_2_pressure",
            "inlane_2_num_vehicle",
            "inlane_2_num_waiting_vehicle",
            "inlane_2_sum_waiting_time",
            "inlane_2_delay",
            "inlane_2_pressure",
            "inter_2_current_phase"
        ], 'Should be the same with TinyLight. '

        self.num_in_lane = len(self.inter.n_in_lane_id)
        self.n_input_feature_dim = [self.num_phase] * 5 + [self.num_in_lane] * 5 + [self.num_phase]
        self.n_layer_1_dim = self.cur_agent['n_layer_1_dim']
        self.n_layer_2_dim = self.cur_agent['n_layer_2_dim']

        self.activated_path = [
            [random.randint(0, len(self.n_input_feature_dim) - 1), random.randint(0, len(self.n_input_feature_dim) - 1)],
            random.randint(0, len(self.n_layer_1_dim) - 1),
            random.randint(0, len(self.n_layer_2_dim) - 1),
        ]

        self.network_local = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.activated_path,
            self.num_phase
        ).to(self.device)
        self.network_target = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.activated_path,
            self.num_phase
        ).to(self.device)
        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.config[self.config['cur_agent']]['learning_rate']
        )

        copy_model_params(self.network_local, self.network_target)


class _Network(torch.nn.Module):
    def __init__(self, list_n_layer_dim, activated_path, num_phase):
        super(_Network, self).__init__()
        assert len(list_n_layer_dim) == len(activated_path) == 3, 'The following implementation is based on 3-layer TinyLight'
        self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim = list_n_layer_dim
        self.activated_path = activated_path
        self.num_phase = num_phase

        self.input_feature_dim_0 = self.n_input_feature_dim[activated_path[0][0]]
        self.input_feature_dim_1 = self.n_input_feature_dim[activated_path[0][1]]
        self.layer_1_dim = self.n_layer_1_dim[activated_path[1]]
        self.layer_2_dim = self.n_layer_2_dim[activated_path[2]]

        self.net_fea_0 = torch.nn.Sequential(
            torch.nn.Linear(self.input_feature_dim_0, self.layer_1_dim),
            torch.nn.ReLU(),
        )
        self.net_fea_1 = torch.nn.Sequential(
            torch.nn.Linear(self.input_feature_dim_1, self.layer_1_dim),
            torch.nn.ReLU(),
        )
        self.net_tail = torch.nn.Sequential(
            torch.nn.Linear(self.layer_1_dim, self.layer_2_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.layer_2_dim, self.num_phase),
        )

    def forward(self, obs):
        fea_0 = obs[self.activated_path[0][0]]
        fea_1 = obs[self.activated_path[0][1]]

        res_add = self.net_fea_0(fea_0) + self.net_fea_1(fea_1)
        res = self.net_tail(res_add)
        return res
