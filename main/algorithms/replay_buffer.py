import numpy as np
import random

class ReplayBuffer(object):
    # 初始化__init__和长度__len__：
    # _storage
    # _maxsize
    # _next_idx
    def __init__(self, size):
        """Create Prioritized Replay buffer.
            创建优先重放缓冲区。
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
            存储在缓冲区中的最大转换数。当缓冲区溢出时，旧的内存将被删除。
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    #     # 清除操作：
    #     # _storage清空；_next_idx清零
    def clear(self):
        self._storage = []
        self._next_idx = 0
    # 类内的函数method：
    # i. add
    #     如果索引超过当前存储数据的长度，则添加这些数据
    #     如果索引小于当前存储数据的长度，则替换该位置之前的数据
    #     索引值自加1之后对storage maxsize求余，保证该数值不会超过最大存储长度
    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
    # ii. _encode_sample采样记忆库
    #     将每个智能体的数据汇总成一个更长的nparray，这些数据包括：obses_t, actions, rewards, obses_tp1, dones
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)
    # iii. make_index
    #     从存储器中随机抽取batch_size长度的数据
    def make_index(self, batch_size):
        # _storage是存储的数据， 随机数在(0~存储长度)之间产生随机整数
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
    # prioritized Experience Relpay PRE 优先回放
    def PRE(self,batch_size):
        pass
    # iv. make_latest_index:
    #     先从当前Index倒叙排序，一直到maxsize，例如：[_next_idx-1, ..., _next_idx - batch_size - 1]
    #     将上述_encode_sample汇总的list打乱
    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        # 随机打乱数据方法
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)
    # v. sample
    #     给定idxes，从storage中抽取这些数据
    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes)
    # vi. collect
    #     return sample(-1)
    #     指的是收集这些数据的阶段
    def collect(self):
        return self.sample(-1)
