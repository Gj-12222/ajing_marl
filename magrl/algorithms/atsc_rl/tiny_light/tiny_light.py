import os

from algorithms.Trainer.baseline import BaseDRL
import torch
import torch.optim as optim
from .alpha import Alpha
from algorithms.Trainer.baseline.utils import copy_model_params


class TinyLight(BaseDRL):
    name = 'TinyLight'
    def __init__(self, config, algo_config, idx, agents):
        super(TinyLight, self).__init__(config, algo_config, idx, agents)
        assert self.cur_agent['obs_list'] == [
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
        ], 'If the feature candidates are changed, adjust the following self.n_input_feature_dim accordingly. '

        self.num_in_lane = len(self.inter.n_in_lane_id)
        self.n_input_feature_dim = [self.num_phase] * 5 + [self.num_in_lane] * 5 + [self.num_phase] * 1  # [4,4,4,4,4,18,18,18,18,18,4]
        self.n_layer_1_dim = self.cur_agent['n_layer_1_dim']  # [16, 18, 20, 22, 24],
        self.n_layer_2_dim = self.cur_agent['n_layer_2_dim']  # [16, 18, 20, 22, 24],
        self.n_alpha = torch.nn.ModuleList([
            Alpha(elem_size=len(self.n_input_feature_dim), config=self.config),
            Alpha(elem_size=len(self.n_layer_1_dim), config=self.config),
            Alpha(elem_size=len(self.n_layer_2_dim), config=self.config)
        ])

        self.network_local = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.n_alpha,
            self.num_phase,
        ).to(self.device)
        self.network_target = _Network(
            [self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim],
            self.n_alpha,
            self.num_phase,
        ).to(self.device)

        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent['learning_rate']
        )
        self.alpha_optim = optim.RMSprop(
            self.n_alpha.parameters(),
            lr=self.cur_agent['learning_rate']
        )
        self.network_lr_scheduler = optim.lr_scheduler.StepLR(
            self.network_optim,
            step_size=10,
            gamma=0.5
        )

        self.beta = 16.0  # weight of alpha regularizer
        copy_model_params(self.network_local, self.network_target)

    def preUpdate(self):
        self.sample_indexs = None

    def update(self, agents, train_step=None, agent_index=None):
        if self._time_to_learn():
            obs, act, rew, next_obs, done = self.replay_buffer.sample_experience()

            # update alpha
            if any([not alpha.is_frozen for alpha in self.n_alpha]):
                alpha_loss = self._compute_alpha_loss(obs, act, rew, next_obs, done)
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
            else:
                alpha_loss = 0

            # update local network
            try:
                network_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
                self.network_optim.zero_grad()
                network_loss.backward()
                self.network_optim.step()
            except:
                try:
                    with torch.autograd.detect_anomaly():
                        network_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
                        self.network_optim.zero_grad()
                        network_loss.backward()
                        self.network_optim.step()
                except:
                    network_loss = 0
                    print('error! but continue!')
            # update target network
            for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
            return {"alpha_loss": alpha_loss,
                    "network_loss": network_loss}
        else:
            return {"alpha_loss": 0,
                    "network_loss": 0}

    def _compute_alpha_loss(self, obs, act, rew, next_obs, done):
        critic_loss = self._compute_critic_loss(obs, act, rew, next_obs, done)
        alpha_loss = critic_loss
        for alpha in self.n_alpha:
            ent = alpha.get_entropy()
            alpha_loss += self.beta * ent
        return alpha_loss

    def hard_threshold_and_freeze_alpha(self):
        self.n_alpha[0].hard_threshold_and_freeze_alpha(2)
        self.n_alpha[1].hard_threshold_and_freeze_alpha(1)
        self.n_alpha[2].hard_threshold_and_freeze_alpha(1)

    def get_alpha_desc(self):
        desc = 'alpha of inter {}: '.format(self.inter.inter_idx)
        for alpha in self.n_alpha:
            desc += '\n{}'.format(alpha.get_desc())
        return desc

    def train(self):
        self.network_local.train()
        self.n_alpha.train()
        self.on_training = True
    def eval(self):
        self.network_local.eval()
        self.n_alpha.eval()
        self.on_training = False

    def save_model(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save({
            'network_local': self.network_local.state_dict(),
            'network_target': self.network_target.state_dict(),
            'n_alpha': self.n_alpha.state_dict(),
            'n_is_frozen': [alpha.is_frozen for alpha in self.n_alpha],
            'n_alive_idx': [alpha.get_alive_idx() for alpha in self.n_alpha],
        }, model_path)

    def load_model(self, model_path):
        model_dict = torch.load(model_path)
        self.network_local.load_state_dict(model_dict['network_local'])
        self.network_target.load_state_dict(model_dict['network_target'])
        self.n_alpha.load_state_dict(model_dict['n_alpha'])
        for alpha_idx, alpha in enumerate(self.n_alpha):
            alpha.is_frozen = model_dict['n_is_frozen'][alpha_idx]
            alpha.n_alive_idx_after_frozen = model_dict['n_alive_idx'][alpha_idx]


class _Network(torch.nn.Module):
    def __init__(self, list_n_layer_dim, n_alpha, num_phase):
        super(_Network, self).__init__()
        assert len(list_n_layer_dim) == 3
        self.n_input_feature_dim, self.n_layer_1_dim, self.n_layer_2_dim = list_n_layer_dim
        self.n_alpha = n_alpha
        self.num_phase = num_phase

        self.input_2_first_layer = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.n_input_feature_dim[idx_input], self.n_layer_1_dim[idx_layer_1]), # [4, 16]
                    torch.nn.ReLU(),
                )
                for idx_layer_1 in range(len(self.n_layer_1_dim))  # 5层 [16, 18, 20, 22, 24],
            ])
            for idx_input in range(len(self.n_input_feature_dim)) # 每个 feature
        ])

        self.first_layer_2_second_layer = torch.nn.ModuleList([
            torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(self.n_layer_1_dim[idx_layer_1], self.n_layer_2_dim[idx_layer_2]),
                    torch.nn.ReLU(),
                )
                for idx_layer_2 in range(len(self.n_layer_2_dim))  # 第2层 5个块 [16, 18, 20, 22, 24],
            ])
            for idx_layer_1 in range(len(self.n_layer_1_dim))  # # 每个 feature 的 第1层每个块 [16, 18, 20, 22, 24],
        ])

        self.second_layer_2_last_layer = torch.nn.ModuleList([
            torch.nn.Linear(self.n_layer_2_dim[idx_layer_2], self.num_phase)  # [16, 4]
            for idx_layer_2 in range(len(self.n_layer_2_dim)) # 第2层每个块 [16, 18, 20, 22, 24],
        ])
        # 每个 feature是 5个 torch.nn.ModuleList([
        #                   torch.nn.Sequential(
        #                     torch.nn.Linear(self.n_input_feature_dim[idx_input], self.n_layer_1_dim[idx_layer_1]), # [4, 16]
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(self.n_layer_1_dim[idx_layer_1], self.n_layer_2_dim[idx_layer_2]),
        #                     torch.nn.ReLU(),
        #                     torch.nn.Linear(self.n_layer_2_dim[idx_layer_2], self.num_phase))
        #                     ])
    def forward(self, obs):
        res_first_layer = [None for _ in range(len(self.n_layer_1_dim))]
        res_second_layer = [None for _ in range(len(self.n_layer_2_dim))]
        res_last_layer = None

        if not self.n_alpha[0].is_frozen:
            # BEFORE frozen: paths are weighted by the alpha term
            norm_alpha_0 = self.n_alpha[0].get_softmax_value()
            for idx_feature in range(len(self.n_input_feature_dim)):
                for jdx_layer_1 in range(len(self.n_layer_1_dim)):
                    elem = self.input_2_first_layer[idx_feature][jdx_layer_1](obs[idx_feature]) * norm_alpha_0[idx_feature]
                    if res_first_layer[jdx_layer_1] is None:
                        res_first_layer[jdx_layer_1] = elem
                    else:
                        res_first_layer[jdx_layer_1] += elem

            norm_alpha_1 = self.n_alpha[1].get_softmax_value()
            for idx_layer_1 in range(len(self.n_layer_1_dim)):
                for jdx_layer_2 in range(len(self.n_layer_2_dim)):
                    elem = self.first_layer_2_second_layer[idx_layer_1][jdx_layer_2](res_first_layer[idx_layer_1]) * norm_alpha_1[idx_layer_1]
                    if res_second_layer[jdx_layer_2] is None:
                        res_second_layer[jdx_layer_2] = elem
                    else:
                        res_second_layer[jdx_layer_2] += elem

            norm_alpha_2 = self.n_alpha[2].get_softmax_value()
            for idx_layer_2 in range(len(self.n_layer_2_dim)):
                elem = self.second_layer_2_last_layer[idx_layer_2](res_second_layer[idx_layer_2]) * norm_alpha_2[idx_layer_2]
                if res_last_layer is None:
                    res_last_layer = elem
                else:
                    res_last_layer += elem
        else:
            # AFTER frozen: only alive paths are activated
            n_alive_alpha_0 = self.n_alpha[0].get_alive_idx()
            n_alive_alpha_1 = self.n_alpha[1].get_alive_idx()
            n_alive_alpha_2 = self.n_alpha[2].get_alive_idx()

            for idx_feature in n_alive_alpha_0:
                for jdx_layer_1 in n_alive_alpha_1:
                    elem = self.input_2_first_layer[idx_feature][jdx_layer_1](obs[idx_feature])
                    if res_first_layer[jdx_layer_1] is None:
                        res_first_layer[jdx_layer_1] = elem
                    else:
                        res_first_layer[jdx_layer_1] += elem

            for idx_layer_1 in n_alive_alpha_1:
                for jdx_layer_2 in n_alive_alpha_2:
                    elem = self.first_layer_2_second_layer[idx_layer_1][jdx_layer_2](res_first_layer[idx_layer_1])
                    if res_second_layer[jdx_layer_2] is None:
                        res_second_layer[jdx_layer_2] = elem
                    else:
                        res_second_layer[jdx_layer_2] += elem

            for idx_layer_2 in n_alive_alpha_2:
                elem = self.second_layer_2_last_layer[idx_layer_2](res_second_layer[idx_layer_2])
                if res_last_layer is None:
                    res_last_layer = elem
                else:
                    res_last_layer += elem

        return res_last_layer


