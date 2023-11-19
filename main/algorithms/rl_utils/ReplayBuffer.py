r'''

store experience rl in environment


'''

import numpy as np
import random
from algorithms.rl_utils.batch import Batch
import copy

class SampleIndexs:
    sample_indexs = None

# off-policy
class ReplayBufferTransition(object):
    # _storage
    # _maxsize
    # _next_idx
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = Batch()
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return self._storage.shape[0]

    def clear(self):
        self._storage = Batch()
        self._next_idx = 0

    # i. add
    def add(self, data):
        r''' store generated data'''
        # process data.shape
        shape = data.shape

        if len(shape) < 2:
            for name in data.keys():
                data[name] = data[name][None]

        if len(shape) > 2:
            for name in data.keys():
                data[name] = data[name].reshape(shape[0] * shape[1], -1)

        try:
            length = len(self._storage)
        except:
            length = 0
        if self._next_idx >= length:  # 未存满
            if self._next_idx + 1 > self._maxsize:
                split_index = self._maxsize - self._next_idx  # 5000 - 4960 = 40
                self._storage.cat_(data[:split_index])  # 40
                self._storage[:split_index] = data[-split_index:]  # 60 - 40 = 20
            else:
                self._storage.cat_(data)
        else:  # 存满了开始覆盖
            if self._next_idx + 1 > self._maxsize:  # 如果不能一次覆盖完，要拆开覆盖  4960 50  5000
                split_index = self._maxsize - self._next_idx  # 5000 - 4960 = 40
                self._storage[self._next_idx:] = data[:split_index]  # 4960-5000   0-40
                self._storage[:split_index] = data[-split_index:]  # 60 - 40 = 20
            else:  # 能一次覆盖完
                self._storage[self._next_idx: self._next_idx +1] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize  # [ 0, 5000]

    # ii. _encode_sample
    def _encode_sample(self, idxes):
        return self._storage[idxes]

    # iii. make_index
    def make_index(self, batch_size):
        return np.random.randint(0, len(self._storage) - 1, batch_size)

    # iv. make_latest_index:
    def make_latest_index(self, batch_size):
        # idx = np.mod(np.arange(self._next_idx - 1, self._next_idx - 1 - batch_size, -1), self._maxsize)
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    # v. sample
    def sample(self, batch_size):
        """ Sample a batch of experiences. """
        idxes = self.make_index(batch_size)
        return self._encode_sample(idxes)

    # vi. collect
    def collect(self):
        return self._storage

    def _keys(self):
        return list(self._storage.keys())


# on-policy mappo /share
class ReplayBufferTrajectory(object):
    # _storage
    # _maxsize
    # _next_idx
    def __init__(self, agents, agent_index, cfg):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.agents = agents
        self.agent_index = agent_index
        self.env = agents[agent_index].env
        self.cfg = cfg

        # self.n_rollout_threads = cfg['n_rollout_threads']  并行env数量
        self.rnn_hidden_dim = cfg['hidden_dim']
        self.rnn_layer_dim = cfg['rnn_layer_dim']
        self.gamma = cfg['gamma']
        self.gae_lambda = cfg['gae_lambda']
        self._use_gae = cfg['use_gae']
        self._use_popart = cfg['use_popart']
        self._use_valuenorm = cfg['use_valueNorm']
        self._use_proper_time_limits = cfg['use_proper_time_limits']
        self._use_rnn_policy = cfg['use_rnn_policy']
        self.obs_dim = self.agents[agent_index].obs_dim
        self.action_type = self.agents[agent_index].action_type
        self.action_dim = self.agents[agent_index].action_dim
        self.join_obs_dim = sum([self.agents[i].obs_dim for i in range(len(self.agents))])

        self.buffer_len = int(cfg['max_episode_len'] / self.env.agent_cfg['action_interval'])
        self._maxsize = self.buffer_len

        #  pre-create buffer data
        # [B, F]
        self.data_dict = {'obs': np.zeros((self.buffer_len + 1, self.obs_dim), dtype=np.float32),
                          # 'join_obs': np.zeros((1, self.join_obs_dim), dtype=np.float32),
                          'old_value': np.zeros((self.buffer_len + 1, 1), dtype=np.float32),
                          'returns': np.zeros((self.buffer_len + 1, 1), dtype=np.float32),
                          'available_actions': np.zeros((self.buffer_len + 1, 1), dtype=np.float32)
                          if self.action_type == 'discrete' else None,
                          'action': np.zeros((self.buffer_len, 1), dtype=np.float32),
                          'action_log_prob': np.zeros((self.buffer_len, 1), dtype=np.float32),
                          'reward': np.zeros((self.buffer_len, 1), dtype=np.float32),
                          'mask': np.ones((self.buffer_len + 1, 1), dtype=np.float32),
                          'bad_mask': np.ones((self.buffer_len + 1, 1), dtype=np.float32),
                          'active_mask': np.ones((self.buffer_len + 1, 1), dtype=np.float32)}
        # [B, L, H]
        self.rnn_dict = {'actor_rnn_hidden_state': np.zeros(
            (self.buffer_len + 1, self.rnn_layer_dim, self.rnn_hidden_dim),
            dtype=np.float32),
            'critic_rnn_hidden_state': np.zeros(
                (self.buffer_len + 1, self.rnn_layer_dim, self.rnn_hidden_dim),
                dtype=np.float32)}

        self._storage = Batch(self.data_dict, copy=True)
        self._rnn_storage = Batch(self.rnn_dict, copy=True)

        self._next_idx = 0

    def __len__(self):
        return self._next_idx

    def clear(self):
        self._storage = Batch(self.data_dict, copy=True)
        self._rnn_storage = Batch(self.rnn_dict, copy=True)

        self._next_idx = 0

    # i. add
    def add(self, data, rnn_data=None):
        r''' store generated data'''
        self._storage.obs[self._next_idx + 1] = data.obs.copy()
        self._storage.action[self._next_idx] = data.action.copy()
        self._storage.action_log_prob[self._next_idx] = data.action_log_prob.copy()
        self._storage.old_value[self._next_idx] = data.old_value.copy()
        self._storage.reward[self._next_idx] = data.reward.copy()
        self._storage.mask[self._next_idx + 1] = data.mask.copy()
        if 'bad_mask' in data:
            self._storage.bad_mask[self._next_idx + 1] = data.bad_mask.copy()
        if 'active_mask' in data:
            self._storage.active_mask[self._next_idx + 1] = data.active_mask.copy()
        if 'available_action' in data:
            self._storage.available_action[self._next_idx + 1] = data.available_action.copy()

        if not rnn_data is None:
            if (data.done==True).sum() > 0:
                rnn_data.actor_rnn_hidden_state[data.done.squeeze(axis=0) == True] = \
                    np.zeros(((data.done==True).sum(), self.rnn_layer_dim, self.rnn_hidden_dim), dtype=np.float32)
                rnn_data.critic_rnn_hidden_state[data.done.squeeze(axis=0) == True] = \
                    np.zeros(((data.done == True).sum(), self.rnn_layer_dim, self.rnn_hidden_dim), dtype=np.float32)

            self._rnn_storage.actor_rnn_hidden_state[self._next_idx + 1] = rnn_data.actor_rnn_hidden_state.copy().squeeze(axis=0)
            self._rnn_storage.critic_rnn_hidden_state[self._next_idx + 1] = rnn_data.critic_rnn_hidden_state.copy().squeeze(axis=0)

        self._next_idx = (self._next_idx + 1) % self._maxsize  # [ 0, 5000]

    # ii. get_data
    def get_data(self, _copy=True):
        if _copy == True:
            get_data, rnn_data = copy.deepcopy(self._storage) ,copy.deepcopy(self._rnn_storage)
        else:
            get_data, rnn_data = self._storage, self._rnn_storage

        return get_data, rnn_data

    # iii. make_index
    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    # iv. after_update:
    def after_update(self):
        self._storage.obs[0] = self._storage.obs[-1].copy()
        self._storage.mask[0] = self._storage.mask[-1].copy()
        self._storage.bad_mask[0] = self._storage.bad_mask[-1].copy()
        self._storage.active_mask[0] = self._storage.active_mask[-1].copy()
        if self._storage.available_actions is not None:
            self._storage.available_actions[0] = self._storage.available_actions[-1].copy()

        self._rnn_storage.actor_rnn_hidden_state[0] = self._rnn_storage.actor_rnn_hidden_state[-1].copy()
        self._rnn_storage.critic_rnn_hidden_state[0] = self._rnn_storage.critic_rnn_hidden_state[-1].copy()

    # v. compute_return
    def compute_return(self, next_value, value_normalizer):
        if self._use_proper_time_limits:
            if self._use_gae:
                self._storage.old_value[-1] = next_value
                gae = 0
                for step in reversed(range(self._storage.reward.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self._storage.reward[step] + self.gamma * value_normalizer.denormalize(self._storage.old_value[
                            step + 1]) * self._storage.mask[step + 1] - value_normalizer.denormalize(self._storage.old_value[step])
                        gae = delta + self.gamma * self.gae_lambda * self._storage.mask[step + 1] * gae
                        gae = gae * self._storage.bad_mask[step + 1]
                        self._storage.returns[step] = gae + value_normalizer.denormalize(self._storage.old_value[step])
                    else:
                        delta = self._storage.reward[step] + self.gamma * self._storage.old_value[step + 1] * self._storage.mask[step + 1] - self._storage.old_value[step]
                        gae = delta + self.gamma * self.gae_lambda * self._storage.mask[step + 1] * gae
                        gae = gae * self._storage.bad_mask[step + 1]
                        self._storage.returns[step] = gae + self._storage.old_value[step]
            else:
                self._storage.returns[-1] = next_value
                for step in reversed(range(self._storage.reward.shape[0])):
                    if self._use_popart:
                        self._storage.returns[step] = (self._storage.returns[step + 1] * self.gamma * self._storage.mask[step + 1] + self._storage.reward[step]) * self._storage.bad_mask[step + 1] \
                            + (1 - self._storage.bad_mask[step + 1]) * value_normalizer.denormalize(self._storage.old_value[step])
                    else:
                        self._storage.returns[step] = (self._storage.returns[step + 1] * self.gamma * self._storage.mask[step + 1] + self._storage.reward[step]) * self._storage.bad_mask[step + 1] \
                            + (1 - self._storage.bad_mask[step + 1]) * self._storage.old_value[step]
        else:
            if self._use_gae:
                self._storage.old_value[-1] = next_value
                gae = 0
                for step in reversed(range(self._storage.reward.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self._storage.reward[step] + self.gamma * \
                                value_normalizer.denormalize(self._storage.old_value[step + 1]) * \
                                self._storage.mask[step + 1] - value_normalizer.denormalize(self._storage.old_value[step])
                        gae = delta + self.gamma * self.gae_lambda * self._storage.mask[step + 1] * gae
                        self._storage.returns[step] = gae + value_normalizer.denormalize(self._storage.old_value[step])
                    else:
                        delta = self._storage.reward[step] + self.gamma * self._storage.old_value[step + 1] * \
                                self._storage.mask[step + 1] - self._storage.old_value[step]
                        gae = delta + self.gamma * self.gae_lambda * self._storage.mask[step + 1] * gae
                        self._storage.returns[step] = gae + self._storage.old_value[step]
            else:
                self._storage.returns[-1] = next_value
                for step in reversed(range(self._storage.reward.shape[0])):
                    self._storage.returns[step] = self._storage.returns[step + 1] * \
                                                  self.gamma * self._storage.mask[step + 1] + self._storage.reward[step]



    # vi. collect
    def collect(self):
        return self.sample(-1)


# Prioritized Experience Replay PER
class PER(object):
    pass


if __name__ == '__main__':
    replay_buffer = ReplayBufferTransition(5000)
    data = {'action': np.ones(256, 3)}
    data = Batch(data)
    for _ in range(10000):
        replay_buffer.add(data)
        print(len(replay_buffer))
    for _ in range(100):
        batch_index = replay_buffer.make_index(512)
        data = replay_buffer.sample_index(batch_index)
