from .. import BaseDRL
import torch
import torch.optim as optim
from ..utils import copy_model_params


class FRAP(BaseDRL):
    """
    Zheng, Guanjie, et al. "Learning phase competition for traffic signal control."
        Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
    """
    def __init__(self, config, env, idx):
        super(FRAP, self).__init__(config, env, idx)
        self.phase_2_passable_lanelink = torch.tensor(self.inter.phase_2_passable_lanelink_idx)
        self.network_local = _Network(
            sum(self.inter.n_num_lanelink),
            len(self.inter.n_phase),
            self.phase_2_passable_lanelink
        )
        self.network_target = _Network(
            sum(self.inter.n_num_lanelink),
            len(self.inter.n_phase),
            self.phase_2_passable_lanelink
        )
        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent['learning_rate']
        )
        copy_model_params(self.network_local, self.network_target)


class _Network(torch.nn.Module):
    def __init__(self, num_lanelink, num_phase, phase_2_passable_lanelink):
        super(_Network, self).__init__()
        self.num_lanelink = num_lanelink
        self.num_phase = num_phase
        self.phase_2_passable_lanelink = phase_2_passable_lanelink
        self.lanelink_2_applicable_phase = phase_2_passable_lanelink.permute(1, 0)  # num_lanelink * num_phase
        self.phase_competition_mask = self._get_phase_competition_mask(phase_2_passable_lanelink)  # num_phase * (num_phase - 1)
        self.dim_embedding = 4
        self.dim_hidden_repr = 16
        self.dim_conv_repr = 20  # section 4.3.5

        self.phase_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Ws, bs in Eq. 3
        self.num_vehicle_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=1, out_features=self.dim_embedding),
            torch.nn.ReLU(),
        )  # Wv, bv in Eq. 3
        self.lanelink_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.dim_embedding * 2, out_features=self.dim_hidden_repr),
            torch.nn.ReLU(),
        )  # Wh, bh in Eq. 4

        self.relation_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.dim_embedding)  # sec. 4.3.3

        self.conv_cube = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_hidden_repr * 2, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.conv_relation = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_embedding, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
        )
        self.tail_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=self.dim_conv_repr, kernel_size=(1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=self.dim_conv_repr, out_channels=1, kernel_size=(1, 1)),
        )

    def forward(self, obs):
        batch_size = obs[0].shape[0]
        b_current_passable_lanelink = torch.matmul(obs[0].float(), self.phase_2_passable_lanelink.float())  # batch * num_lanelink
        b_num_waiting_vehicle = obs[1]  # batch * num_lanelink

        # Eq. 3
        lanelink_embedding_wrt_phase = self.phase_embedding(b_current_passable_lanelink.unsqueeze(-1))  # batch * num_lanelink * embedding_dim
        lanelink_embedding_wrt_num_vehicle = self.num_vehicle_embedding(b_num_waiting_vehicle.unsqueeze(-1))  # batch * num_lanelink * embedding_dim
        # Eq. 4
        lanelink_embedding = torch.cat([lanelink_embedding_wrt_phase, lanelink_embedding_wrt_num_vehicle], dim=2)  # batch * num_lanelink * (2 *embedding_dim)
        lanelink_embedding = self.lanelink_embedding(lanelink_embedding)  # batch * num_lanelink * dim_hidden_repr
        lanelink_embedding = lanelink_embedding.permute(0, 2, 1)  # batch * dim_hidden_repr * num_lanelink

        # Eq. 5
        phase_embedding = torch.matmul(lanelink_embedding, self.lanelink_2_applicable_phase.float())  # batch * dim_hidden_repr * num_phase
        phase_embedding = phase_embedding.permute(0, 2, 1)  # batch * num_phase * dim_hidden_repr
        phase_embedding_cube = self._get_phase_embedding_cube(batch_size, phase_embedding)  # batch * (dim_hidden_repr * 2) * num_phase * (num_phase - 1)
        # Eq. 6
        phase_conv = self.conv_cube(phase_embedding_cube)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 7
        relation_embedding = self.relation_embedding(self.phase_competition_mask).permute(2, 0, 1).unsqueeze(0)  # (batch=1) * embedding_dim * num_phase * (num_phase - 1)
        relation_conv = self.conv_relation(relation_embedding)  # batch * dim_conv_repr * num_phase * (num_phase - 1)

        # Eq. 8
        combined_feature = phase_conv * relation_conv  # batch * dim_conv_repr * num_phase * (num_phase - 1)
        before_merge = self.tail_layer(combined_feature)  # batch * 1 * num_phase * (num_phase - 1)
        q_values = torch.sum(before_merge, dim=3).squeeze(1)  # batch * num_phase
        return q_values

    def _get_phase_embedding_cube(self, batch_size, phase_embedding):
        phase_embedding_cube = torch.zeros((batch_size, 32, self.num_phase, self.num_phase - 1))
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                phase_embedding_comb = torch.cat([
                    phase_embedding[:, phase_idx, :],
                    phase_embedding[:, phase_jdx, :]
                ], dim=1)
                phase_embedding_cube[:, :, phase_idx, continuous_jdx] = phase_embedding_comb
                continuous_jdx += 1
        return phase_embedding_cube

    def _get_phase_competition_mask(self, phase_2_passable_lanelink):
        mask = torch.zeros((self.num_phase, self.num_phase - 1), dtype=torch.int64)
        for phase_idx in range(self.num_phase):
            continuous_jdx = 0
            for phase_jdx in range(self.num_phase):
                if phase_idx == phase_jdx:
                    continue
                for lanelink_idx in range(self.num_lanelink):
                    if phase_2_passable_lanelink[phase_idx][lanelink_idx] == phase_2_passable_lanelink[phase_jdx][lanelink_idx] == 1:
                        mask[phase_idx][continuous_jdx] = 1
                continuous_jdx += 1
        return mask
