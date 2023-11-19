import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, obs_size, device):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.obs_size = obs_size
        self.device = device

        self.memory = {
            'obs': self._get_obs_placeholder(),
            'act': np.empty((self.buffer_size, 1), dtype=np.int64),
            'rew': np.empty((self.buffer_size, 1), dtype=np.float32),
            'next_obs': self._get_obs_placeholder(),
            'done': np.empty((self.buffer_size, 1), dtype=np.bool)
        }
        self._cur_idx = 0
        self.current_size = 0

    def _get_obs_placeholder(self):
        if isinstance(self.obs_size, list):
            return [np.empty((self.buffer_size, *siz), dtype=np.float32) for siz in self.obs_size]
        else:
            return np.empty((self.buffer_size, *self.obs_size), dtype=np.float32)

    def dump(self):
        return {
            "memory": self.memory,
            "_cur_idx": self._cur_idx,
            "current_size": self.current_size
        }

    def load(self, obj):
        self.memory = obj["memory"]
        self._cur_idx = obj["_cur_idx"]
        self.current_size = obj["current_size"]

    def reset(self):
        self._cur_idx = 0
        self.current_size = 0

    def store_experience(self, obs, act, rew, next_obs, done):
        if isinstance(self.obs_size, list):
            for feature_idx, ith_obs in enumerate(obs):
                self.memory['obs'][feature_idx][self._cur_idx] = ith_obs
            for feature_idx, ith_next_obs in enumerate(next_obs):
                self.memory['next_obs'][feature_idx][self._cur_idx] = ith_next_obs
        else:
            self.memory['obs'][self._cur_idx] = obs
            self.memory['next_obs'][self._cur_idx] = next_obs

        self.memory['act'][self._cur_idx] = act
        self.memory['rew'][self._cur_idx] = rew
        self.memory['done'][self._cur_idx] = done

        self.current_size = min(self.current_size + 1, self.buffer_size)
        self._cur_idx = (self._cur_idx + 1) % self.buffer_size

    def sample_experience(self, batch_size=None, idxs=None):
        batch_size = batch_size or self.batch_size
        if idxs is None:
            idxs = np.random.choice(self.current_size, batch_size, replace=True)

        if isinstance(self.obs_size, list):
            obs, next_obs = [], []
            for obs_feature_idx in range(len(self.obs_size)):
                obs.append(self._to_torch(self.memory['obs'][obs_feature_idx][idxs]))
                next_obs.append(self._to_torch(self.memory['next_obs'][obs_feature_idx][idxs]))
        else:
            obs = self._to_torch(self.memory['obs'][idxs])
            next_obs = self._to_torch(self.memory['next_obs'][idxs])

        act = self._to_torch(self.memory['act'][idxs])
        rew = self._to_torch(self.memory['rew'][idxs])
        done = self._to_torch(self.memory['done'][idxs])
        return obs, act, rew, next_obs, done

    def get_sample_indexes(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        return np.random.choice(self.current_size, batch_size, replace=True)

    def _to_torch(self, np_elem):
        return torch.from_numpy(np_elem).to(self.device)

    def __str__(self):
        return str("current size: {} / {}".format(self.current_size, self.buffer_size))
