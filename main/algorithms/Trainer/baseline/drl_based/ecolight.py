from .. import BaseDRL
import torch
import torch.optim as optim
from ..utils import copy_model_params
import random


class EcoLight(BaseDRL):
    """
    Chauhan, Sachin, Kashish Bansal, and Rijurekha Sen. "EcoLight: Intersection Control in Developing Regions Under
        Extreme Budget and Network Constraints." Advances in Neural Information Processing Systems 33 (2020).

    Note: EcoLight has several simplification based on the realistic situation in Delhi. This implementation is
        based on the original version (defined in Section 3 and the first paragraph of Section 4), which
        represents the best performance of EcoLight (according to the authors' experimental results).
    """
    def __init__(self, config, env, idx):
        super(EcoLight, self).__init__(config, env, idx)
        self.phase_2_passable_lane = torch.tensor(self.inter.phase_2_passable_lane_idx)
        self.network_local = _Network(self.phase_2_passable_lane)
        self.network_target = _Network(self.phase_2_passable_lane)
        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent["learning_rate"]
        )
        self.network_lr_scheduler = optim.lr_scheduler.StepLR(self.network_optim, 20, 0.1)
        copy_model_params(self.network_local, self.network_target)

    def pick_action(self, n_obs, on_training):
        obs = n_obs[self.idx]
        assert obs[0].shape[0] == 1, 'should be mini-batch with size 1'
        self.network_local.eval()
        with torch.no_grad():
            b_q_value = self.network_local(obs)
            action = torch.argmax(b_q_value, dim=1).item()
            if on_training:
                if random.random() < self.cur_agent["epsilon"]:  # explore
                    action = random.randint(0, 1)
        self.network_local.train()
        if action == 1:
            self.current_phase = (self.current_phase + 1) % self.num_phase
        return self.current_phase

    def store_experience(self, obs, action, reward, next_obs, done):
        current_phase = torch.argmax(obs[0]).item()
        binary_action = int(action != current_phase)  # 0: stay, 1: change
        self.replay_buffer.store_experience(obs, binary_action, reward, next_obs, done)


class _Network(torch.nn.Module):
    def __init__(self, phase_2_passable_lane):
        super(_Network, self).__init__()
        self.phase_2_passable_lane = phase_2_passable_lane.float()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_features=2, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=2)
        )

    def forward(self, obs):
        # obs[0]: current phase, B * num_phase
        # obs[1]: number of cars in each lane, B * num_lane
        b_passable_lane = torch.matmul(obs[0], self.phase_2_passable_lane)  # B * num_lane
        b_num_car_with_green_signal = torch.sum(b_passable_lane * obs[1], dim=1, keepdim=True)  # B * 1
        b_num_car_with_red_signal = torch.sum((1. - b_passable_lane) * obs[1], dim=1, keepdim=True)  # B * 1

        b_feature = torch.cat([b_num_car_with_green_signal, b_num_car_with_red_signal], dim=1)  # B * 2
        q_values = self.net(b_feature)  # B * 2
        return q_values
