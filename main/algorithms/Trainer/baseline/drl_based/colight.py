from .. import BaseDRL
import torch
import torch.optim as optim
import random
import torch.nn.functional as F
from ..utils import copy_model_params
from Env.CityFlowEnv import CityFlowWorld


class CoLight(BaseDRL):
    """
    Wei, Hua, et al. "Colight: Learning network-level cooperation for traffic signal control."
        Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.

    Note: CoLight paper has two definitions for the neighboring intersection, one is defined by the topological
        relationship and the other is by the geographical distance. Here we choose the former definition as this
        one is consistent for all datasets.
    """
    def __init__(self, config, env, idx):
        super(CoLight, self).__init__(config, env, idx)

        n_neighbor_idx = env.n_intersection[idx].n_neighbor_idx
        self.network_local = _Network(
            len(self.inter.n_lane_id),
            len(self.inter.n_phase),
            env, idx, n_neighbor_idx, is_target_network=False
        )
        self.network_target = _Network(
            len(self.inter.n_lane_id),
            len(self.inter.n_phase),
            env, idx, n_neighbor_idx, is_target_network=True
        )
        self.network_optim = optim.RMSprop(
            self.network_local.parameters(),
            lr=self.cur_agent['learning_rate']
        )
        copy_model_params(self.network_local, self.network_target)

    def pick_action(self, n_obs, on_training):
        assert all(obs[0].shape[0] == 1 for obs in n_obs), 'should be mini-batch with size 1'
        self.network_local.eval()
        with torch.no_grad():
            b_q_value = self.network_local(n_obs)
            action = torch.argmax(b_q_value, dim=1).item()
            if on_training and random.random() < self.cur_agent['epsilon']:  # explore
                action = random.randint(0, self.num_phase - 1)
        self.network_local.train()
        self.current_phase = action
        return self.current_phase

    def learn(self):
        if self._time_to_learn():
            # collect experience
            idxs = self.replay_buffer.get_sample_indexes()
            _, act, rew, _, done = self.replay_buffer.sample_experience(idxs=idxs)
            n_obs, n_next_obs = [], []
            for agent in self.env.n_agent:
                obs, _, _, next_obs, _ = agent.replay_buffer.sample_experience(idxs=idxs)
                n_obs.append(obs)
                n_next_obs.append(next_obs)

            # calculate loss and back propagate
            critic_loss = self._compute_critic_loss(n_obs, act, rew, n_next_obs, done)
            self.network_optim.zero_grad()
            critic_loss.backward()
            self.network_optim.step()
            for to_model, from_model in zip(self.network_target.parameters(), self.network_local.parameters()):
                to_model.data.copy_(self.tau * from_model.data + (1.0 - self.tau) * to_model.data)
            return critic_loss
        else:
            return 0

    def _compute_critic_loss(self, n_obs, act, rew, n_next_obs, done):
        with torch.no_grad():
            q_target_next = self.network_target(n_next_obs)
            q_target = rew + self.gamma * torch.max(q_target_next, dim=1, keepdim=True)[0] * (~done)
        q_expected = self.network_local(n_obs).gather(1, act.long())
        critic_loss = F.mse_loss(q_expected, q_target)
        return critic_loss


class _Network(torch.nn.Module):
    def __init__(self, num_lane, num_phase, env: CityFlowWorld, idx, n_neighbor_idx, is_target_network):
        super(_Network, self).__init__()
        self.num_lane = num_lane
        self.num_phase = num_phase
        self.env = env
        self.idx = idx
        self.n_neighbor_idx = n_neighbor_idx
        self.is_target_network = is_target_network

        self.dim_embedding = 32
        self.dim_attention_head = 32
        self.dim_representation = 32
        self.num_head = 5  # determined by Table 3 in CoLight paper

        if idx == 0:
            # convert input features to embeddings
            self.embedding_mlp = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.num_phase + self.num_lane, out_features=self.dim_embedding),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_embedding),
                torch.nn.ReLU(),
            )  # W_e in Eq. 2; the author's original implementation indicates that it is a two-layer structure

            # attention is calculated based on these heads
            self.n_agent_embedding_2_head = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_attention_head),
                    torch.nn.ReLU()
                ) for _ in range(self.num_head)
            ])  # W_t in Eq. 6

            self.n_neighbor_embedding_2_head = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_attention_head),
                    torch.nn.ReLU()
                ) for _ in range(self.num_head)
            ])  # W_s in Eq. 6

            # attention is operated on these hidden representations
            self.n_neighbor_embedding_2_hidden_repr = torch.nn.ModuleList([
                torch.nn.Sequential(
                    torch.nn.Linear(in_features=self.dim_embedding, out_features=self.dim_representation),
                    torch.nn.ReLU()
                ) for _ in range(self.num_head)
            ])  # W_c in Eq. 8

            # q value prediction layer
            self.tailing_net = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.dim_representation, out_features=self.dim_representation),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=self.dim_representation, out_features=self.num_phase)
            )  # W_q in Eq. 8
        elif self.is_target_network:  # parameters are shared among intersections in CoLight
            self.embedding_mlp = self.env.n_agent[0].network_target.embedding_mlp
            self.n_agent_embedding_2_head = self.env.n_agent[0].network_target.n_agent_embedding_2_head
            self.n_neighbor_embedding_2_head = self.env.n_agent[0].network_target.n_neighbor_embedding_2_head
            self.n_neighbor_embedding_2_hidden_repr = self.env.n_agent[0].network_target.n_neighbor_embedding_2_hidden_repr
            self.tailing_net = self.env.n_agent[0].network_target.tailing_net
        else:
            self.embedding_mlp = self.env.n_agent[0].network_local.embedding_mlp
            self.n_agent_embedding_2_head = self.env.n_agent[0].network_local.n_agent_embedding_2_head
            self.n_neighbor_embedding_2_head = self.env.n_agent[0].network_local.n_neighbor_embedding_2_head
            self.n_neighbor_embedding_2_hidden_repr = self.env.n_agent[0].network_local.n_neighbor_embedding_2_hidden_repr
            self.tailing_net = self.env.n_agent[0].network_local.tailing_net

    def forward(self, n_obs):
        n_neighbor_idx = self.env.n_intersection[self.idx].n_neighbor_idx

        # Eq. 2
        all_flatten_feature = torch.cat(n_obs[self.idx], dim=1)  # batch * (num_phase + num_lane)
        agent_embedding = self.embedding_mlp(all_flatten_feature)  # batch * dim_embedding

        n_neighbor_embedding = torch.stack([
            self.env.n_agent[neighbor_idx].network_local.embedding_mlp(
                torch.cat(n_obs[neighbor_idx], dim=1)
            )
            for neighbor_idx in n_neighbor_idx
        ], dim=1)  # batch * num_neighbor * dim_embedding

        n_agent_hidden_repr = []
        for head_idx in range(self.num_head):
            # Eq. 6 & 7
            agent_attention_head = self.n_agent_embedding_2_head[head_idx](agent_embedding).unsqueeze(2)  # batch * dim_attention_head * 1
            n_neighbor_attention_head = self.n_neighbor_embedding_2_head[head_idx](n_neighbor_embedding)  # batch * num_neighbor * dim_attention_head

            attention = torch.softmax(torch.bmm(n_neighbor_attention_head, agent_attention_head), dim=1)  # batch * num_neighbor * 1
            attention = attention.permute(0, 2, 1)  # batch * 1 * num_neighbor

            # Eq. 5
            n_neighbor_hidden_repr = self.n_neighbor_embedding_2_hidden_repr[head_idx](n_neighbor_embedding)  # batch * num_neighbor * dim_representation
            agent_hidden_repr = torch.bmm(attention, n_neighbor_hidden_repr)  # batch * 1 * dim_representation
            agent_hidden_repr = agent_hidden_repr.squeeze(1)  # batch * dim_representation
            n_agent_hidden_repr.append(agent_hidden_repr)

        # Eq. 8
        avg_agent_hidden_repr = torch.mean(torch.stack(n_agent_hidden_repr, dim=2), dim=2)  # batch * dim_representation
        q_values = self.tailing_net(avg_agent_hidden_repr)  # batch * num_phase
        return q_values
