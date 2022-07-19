### 导入库

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnn
import random
import copy

import tools.tf_util as U
from tools.distributions import make_pdtype
from algorithms import AgentTrainer

"""
QMIX 作为 是否为离散动作？？？很奇怪，我觉得连续也行，
 这是一个面向合作场景的算法，旨在用超网络拟合单调性的全局Q，缓解CTDE的集中式训练联合维度爆炸问题
 暂时放下

"""


class QMIXAgentTrainer(AgentTrainer):
    def __init__(self, name,
        actor_critic_model,
        obs_space_n,
        act_space_n,
        agent_index,
        args,
        local_q_func = False):
        # ①智能体参数
        self.name = name
        self.n = len(obs_space_n)
        self.agent_index = agent_index
        self.args = args

        # self.register(name='ops',values=args.)
        # ②AC网络输入变量：
        # 2.1 状态占位符
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_space_n[i], name="observation" + str(i)).get())
        warpper_obs_tuple_to_list = list(obs_space_n[agent_index])[0] * 4
        warpper_obs_ph = U.BatchInput([warpper_obs_tuple_to_list + args.num_units],
                                      name="warpper_obs_and_last_hidden_state" + str(self.agent_index)).get()

        # 2.2 建立mix-Q价值网络及计算图
        self.q_train, self.q_targrt_update, self.q_target_values = mix_q_train(
                scope = self.name,
                obs_ph_n = obs_ph_n,
                act_space_n = act_space_n,
                q_index = agent_index,
                q_func = coma_critic_mlp,
                optimizer = tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
                grad_norm_clipping = 0.5,
                local_func = local_q_func,
                num_units = args.num_units)
        # 2.3 建立策略网络及计算图
        self.act, self.p_train, self.p_target_update, self.p_target_act, self.actor_vars = p_train(
                scope = self.name,
                obs_ph_n = obs_ph_n,
                warpper_obs_ph = warpper_obs_ph,
                act_space_n = act_space_n,
                p_index = agent_index,
                p_func = coma_actor_rnn,
                q_func = coma_critic_mlp,
                b_func = coma_critic_mlp,
                optimizer = tf.train.AdamOptimizer(learning_rate=args.lr * 0.1),
                grad_norm_clipping = 0.5,  # 梯度裁剪
                local_func = local_q_func,  # 判断使用Signal-agentRL还是Multi-agentRL
                num_units = args.num_units,
                rnn_time_step = args.rnn_time_step,
                obs_shape = warpper_obs_tuple_to_list)
        # ③记忆库
        self.QMIX_replay_buffer = QMIXPolicyReplayBuffer(self.args.max_replay_buffer)  # max step的 容量
        # 一次批量提取的最大长度max_replay_buffer_len
        # self.max_replay_buffer_len = args.batch_size * args.max_episode_len  # COMA是存满了再批量更新
        self.max_replay_buffer_len = args.max_episode_len  # COMA是存满了再批量更新
        self.replay_sample_index = None
        """权宜之计，在类里加入 hidden_state，以及保存3个历史片段 """
        self.hidden_state = None
        self.history_states = None

    def action(self, obs):
        # 取前 3个 step 的 state + 当前 state
        if self.hidden_state is None:  # 类似初始化
            self.hidden_state = np.zeros((self.args.num_units,))
        self.warpper(obs)  # 组成4个历史片段
        actor_inputs_feature = np.concatenate([self.history_states] + [self.hidden_state])
        [act, act_hidden] = self.act(actor_inputs_feature[None])
        self.hidden_state = act_hidden[0]
        return act[0]

    def warpper(self, obs_t):
        if self.history_states is None:  # 即第一次交互时的输入初始化
            self.history_states = np.array([obs_t] * self.args.rnn_time_step).reshape(-1, )
        else:  # 依次保存历史state
            temp_warpper = self.history_states.reshape(self.args.rnn_time_step, -1)
            temp_warpper = temp_warpper[1:]
            self.history_states = np.concatenate([temp_warpper, obs_t.reshape(1, -1)], axis=0).reshape(-1, )  # 在第0维度拼接

    def learn_warpper(self, agent_obs):
        agent_warpper_obs = []  #
        temp_history_state = np.concatenate([agent_obs[0], agent_obs[0], agent_obs[0], agent_obs[0]], axis=0)
        for i in range(len(agent_obs)):  # batch
            agent_warpper_obs += [temp_history_state.reshape(-1, )]
            temp_history_state = temp_history_state.reshape(self.args.rnn_time_step, -1)
            temp_history_state = temp_history_state[1:]
            temp_history_state = np.concatenate([temp_history_state, agent_obs[i].reshape(1, -1)],
                                                axis=0)  # 第 0 维度拼接 time_step 维拼接
        agent_warpper_obs += [temp_history_state.reshape(-1, )]  # 最后补一个
        return agent_warpper_obs

    def update(self, agent, t):
        pass

class QMIXPolicyReplayBuffer(object):

    def __init__(self, size):
        self._storage = []
        self._maxsize = int(size)
        self._next_index = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_index = 0

    def add(self, obs_t, action, reward, obs_t1, done_mask):  # 加入replay buffer 长度不超过1个 episode
        data = (obs_t, action, reward, obs_t1, done_mask)
        if self._next_index >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_index] = data
        self._next_index = (self._next_index + 1) % self._maxsize

    ###***************************************###
    ###   on-policy-sample
    ###***************************************###
    # 取最大step步长作为采样索引indexs
    def on_policy_make_indexs_sample(self, step):
        return [step]

    # 普通序列采样数据
    def _on_policy_lambda_encode_sampe(self, indexs):
        obs, act, rew, obs_next, mask = [], [], [], [], []
        for i in range(indexs[0]):
            data = self._storage[i]  # 正序就行
            obs_temp, act_temp, rew_temp, obs_next_temp, mask_temp = data
            obs.append(np.array(obs_temp, copy=False))
            act.append(np.array(act_temp, copy=False))
            rew.append(np.array(rew_temp))
            obs_next.append(np.array(obs_next_temp, copy=False))
            mask.append(np.array(mask_temp))
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(mask)

    # 根据indexs执行采样
    def on_poicy_sampe(self, indexs):
        return self._on_policy_lambda_encode_sampe(indexs)

    ###***************************************###
    ###   off-policy-sample  n-step TD(λ)采样
    ###***************************************###

    def make_sort_sample_index(self, batch_size):
        """ .sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
        list.sort(cmp=None, key=None, reverse=False)
        reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。"""
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        list.sort(index_sample)
        return index_sample

    def TD_lambda_sample(self, indexs, step):
        return self._off_policy_lambda_encode_sample(indexs, step)

    def _off_policy_lambda_encode_sample(self, indexs, step):
        obs, act, rew, obs_next, done_mask = [], [], [], [], []
        obs_dim, act_dim, rew_dim, obs_next_dim, done_mask_dim = self._storage[0]
        # 相同维度的0矩阵
        obs_deep_dim = np.zeros_like(obs_dim)
        act_deep_dim = np.zeros_like(act_dim)
        for i in range(len(indexs)):
            obs_temp, act_temp, rew_temp, obs_next_temp, done_mask_temp = [], [], [], [], []
            for j in range(step):
                if indexs[i] + j > len(self._storage) - 1:
                    break
                data = self._storage[indexs[i] + j]
                obs_t, act_t, rew_t, obs_t1, done_mask_t = data
                obs_temp.append(np.array(obs_t, copy=False))
                act_temp.append(np.array(act_t, copy=False))
                rew_temp.append(rew_t)
                obs_next_temp.append(np.array(obs_t1, copy=False))
                done_mask_temp.append(done_mask_t)
                if done_mask_t == 1.0:  # 终止状态则停止读取  beyond max_buffer_list
                    break
                if i < len(indexs) - 1:  # 先判断是否超indexs维度  如果是最后1个
                    if indexs[i] + j + 1 >= indexs[i + 1]:  # 再判断相邻索引
                        break
            # 补齐轨迹序列-当做死亡时的状态
            if len(obs_temp) < step and len(indexs) >= 2:
                for i in range(step - len(obs_temp)):
                    obs_temp.append(np.array(obs_deep_dim))
                    act_temp.append(np.array(act_deep_dim))
                    rew_temp.append(0.0)
                    obs_next_temp.append(np.array(obs_deep_dim))
                    done_mask_temp.append(1.0)

            obs.append(np.array(obs_temp, copy=False))
            act.append(np.array(act_temp, copy=False))
            rew.append(np.array(rew_temp, copy=False))
            obs_next.append(np.array(obs_next_temp, copy=False))
            done_mask.append(np.array(done_mask_temp, copy=False))
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(done_mask)

    ###***************************************###
    ###   off-policy-sample  1-step TD(0)采样
    ###***************************************###
    # 随机采样
    def make_sample_index(self, batch_size):
        """ .sort() 函数用于对原列表进行排序，如果指定参数，则使用比较函数指定的比较函数。
        list.sort(cmp=None, key=None, reverse=False)
        reverse -- 排序规则，reverse = True 降序， reverse = False 升序（默认）。"""
        index_sample = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        # list.sort(index_sample)
        return index_sample

    # 打乱逆序采样
    def make_last_jam_index(self, batch_size):
        indexs = [(self._next_index - i - 1) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(indexs)
        return indexs

    # 倒序取值
    def collect(self):
        return self.sample(-1)

    def sample(self, indexs):
        return self._encode_sample(indexs)

    def _encode_sample(self, indexs):
        obs, act, rew, obs_next, done_mask = [], [], [], [], []
        for i in indexs:
            data = self._storage[i]
            obs_t, act_t, rew_t, obs_t1, done_mask_t = data
            obs.append(np.array(obs_t, copy=False))
            act.append(np.array(act_t, copy=False))
            rew.append(rew_t)
            obs_next.append(np.array(obs_t1, copy=False))
            done_mask.append(done_mask_t)
        return np.array(obs), np.array(act), np.array(rew), np.array(obs_next), np.array(done_mask)

    ###***************************************###
    ###   PER 优先级回放（未完成）
    ###***************************************###
    def PER(self, batch_size):  # 优先采样回放
        pass
    # pass
