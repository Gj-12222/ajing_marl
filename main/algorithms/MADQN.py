"""
MADQN CTCE

"""

from maddpg import AgentTrainer
from maddpg.common import tf_util as U
import tensorflow as tf
from maddpg.common.distributions import make_pdtype
from maddpg.trainer.replay_buffer import ReplayBuffer
import numpy as np

def soft_update_vars(q_vars, q_target_vars,tau):
    soft_update = []
    for q_var, q_target_var in zip(sorted(q_vars,key=lambda v:v.name),sorted(q_target_vars,key=lambda v:v.name)):
        soft_update.append(q_target_var.assign((1 - tau)*q_target_var + tau*q_var))
    soft_update = tf.group(*soft_update)
    return U.function([],[],updates=[soft_update])

def q_train(make_obs_n,
                act_space_n,
                q_model,
                agent_index,
                optimizer,
                gram_normal_clipping=0.5,
                local_q_func=False,
                num_units=64,
                scope="trainers",
                tau=0.001):
    with tf.variable_scope(scope,reuse=False):
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        act_ph_n = [act_pdtype_n[i].sample_placeholder(prepend_shape=[None], name="action" +str(i)) for i in range(len(act_space_n))]

        action_index_ph = tf.placeholder(dtype=tf.int64, shape=[None,2], name="action_index")

        obs_ph_n = make_obs_n

        q_input = obs_ph_n
        q = q_model(q_input,int(act_pdtype_n[agent_index].param_shape()[0]),scope="q_model", num_units=num_units,activation_fn=None)

        q_vars = U.scope_vars(U.absolute_scope_name("q_model"))
        # 获取q值分布

        q_temp = tf.placeholder(dtype=tf.float32, shape=[None],name="q_s_a")

        q_predit = tf.gather_nd(q, action_index_ph)  # 提取对应index下的q

        q_loss = - tf.reduce_mean(tf.square(q_predit - q_temp)) # [1]
        epilon = tf.reduce_mean(q)
        loss = q_loss + 1e-3 * epilon

        optimizers_var = U.minimize_and_clip(optimizer=optimizer,objective=loss,var_list=q_vars,clip_val=gram_normal_clipping)

        q_probs_sample = U.function(inputs=obs_ph_n, outputs=q)

        train = U.function(inputs=obs_ph_n + [action_index_ph] + [q_temp],outputs=loss,updates=[optimizers_var])
        target_q_prb = q_model(q_input, int(act_pdtype_n[agent_index].param_shape()[0]), scope="q_target_model", num_units=num_units, activation_fn=None)
        target_q_max = tf.reduce_max(target_q_prb, axis=1, keep_dims=False)

        q_target_vars = U.scope_vars(U.absolute_scope_name("q_target_model"))

        q_target_soft_update = soft_update_vars(q_vars=q_vars,q_target_vars=q_target_vars,tau=tau)
        q_target_max_values = U.function(inputs=obs_ph_n, outputs=target_q_max)

        return q_probs_sample, train, q_target_soft_update, q_target_max_values



class MADQNAgentTrainer(AgentTrainer):
    def __init__(self, name,
                 adv_model,
                 obs_shape_n,
                 act_space,  # n个act_space
                 agent_index,
                 args,
                 local_q_func=False):

        self.name=name
        self.agent_index = agent_index
        self.n = len(obs_shape_n)
        self.args = args
        self.actor_space_n = act_space

        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        self.q_probs_sample, self.q_train, self.q_update, self.q_target_values = q_train(scope=name,
                make_obs_n=obs_ph_n,
                act_space_n=act_space,
                q_model=adv_model,
                agent_index=agent_index,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 0.05),
                gram_normal_clipping=0.5,
                local_q_func=local_q_func,
                num_units=args.num_units,
                tau=args.tau)

        self.delta = args.delta

        self.dqn_replay_buffer = ReplayBuffer(size=5e6)
        self.dqn_sample = args.batch_size * args.max_episode_len


    def action(self, obs_n):

        if np.random.rand() <= self.delta:  # epsilon-greedy
            actions = [np.random.randint(low=0,high=(self.actor_space_n[self.agent_index].n)-1, dtype=np.int64) ]# (5,1)是动作
        else:
            obs_n = np.array(obs_n)
            q_probs = self.q_probs_sample(obs_n)
            actions = [np.argmax(q_probs)]
        return actions

    def experience(self, obs, act, rew, obs_next, done, terminal):
        self.dqn_replay_buffer.add(obs, act, rew, obs_next, float(done and terminal))

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t):

        if len(self.dqn_replay_buffer) < self.dqn_sample:
            return

        if not t % 10 == 0:
            return

        if self.delta <= 0.001:
            self.delta = 0.001
        elif t % 6000:
            self.delta = self.delta - 0.005

        self.replay_sample_index =self.dqn_replay_buffer.make_index(self.args.batch_size)

        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index

        for i in range(self.n):
            obs, act, _, obs_next, _ = agents[i].dqn_replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)

        _, self_act, rew, _, done = self.dqn_replay_buffer.sample_index(index)

        num_sample = 1
        target_u = 0.0

        for i in range(num_sample):
            q_target_next_max = self.q_target_values(obs_next_n)  # [batch_size,1] = q_next_max
            # target_u = R + gamma * (1 - done——终止条件) * Q
            target_u += rew + self.args.gamma * (1 - done) * q_target_next_max  # [batch_size,1]
        target_u /= num_sample

        action_index_arithmetic_sequence = np.linspace(start=0, stop=len(self_act)-1, num=len(self_act), dtype=np.int64).reshape(-1,1)
        self_act = self_act.reshape(-1, 1)

        action_indexs = np.concatenate((action_index_arithmetic_sequence,self_act), axis=1)

        q_loss = self.q_train(*(obs_n + [action_indexs] + [target_u]))

        self.q_update()

        return [q_loss, np.mean(target_u), np.mean(rew),  np.std(target_u)]












