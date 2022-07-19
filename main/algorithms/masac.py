
"""
core code for the MASAC algorithm
（MASAC算法的核心代码）
自适应alpha (未实现)

    算法流程：
    ① def init 初始化各种参数
        1.1 多智能体的参数，状态，动作等
        1.2 actor-critic网络及所有function函数
        1.3 记忆库
    ② def actor-critic网络
        2.1 actor网咯-
        2.2 critic网咯-
    ③ def 选择动作action
        3.1 根据actor网络初始化顶的选择动作的function，得到网络输出值
    ④ def 记忆库存储
        4.1 存储<s,a,r,s',d>~D
    ⑤ def 网咯更新
        5.1 更新critic_target:
            5.1.1 Q_loss = MSE(Q(s,a;Θ) - Y(s',a';Θ_target))^2   # TD(0) error
                Y = r + (1 - d)*γ*{Q(s'a';Θ_target)|[a'=π(a'|s';φ_target)] - α*ln(π(a'|s;φ_target))}
            5.1.2 π_loss = MSE(q(S,a,θ) - α*ln(π(a|s;φ))  # 最大熵
            5.1.3 软更新
                θ_target <-- Γ*θ+(1 - Γ)*θ_target
                φ_target <-- Γ*φ+(1 - Γ)*φ_target
"""
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tools.tf_util as U
from tools.distributions import make_pdtype
from algorithms.replay_buffer import ReplayBuffer
from algorithms import AgentTrainer

epsilon = 1e-8
"""建立(MA)SACAgentTrainer"""
# Actor-critic
def sac_nn(input,num_outputs,scope,  reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

#  参数更新函数
def make_update_exp(vals, target_vals):
    polyak = 1.0 - (5*1e-3)
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))   #
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

# 对数形式高斯分布- 重新参数化的动作分布概率
def log_gaussian_policy(act_resample, act_mu, act_logstd):
    # log_s = -0.5 *[( x - u)/ (e^log var)]^2 + 2*(log var) + 2*logπ # 高斯分布对数形式
    log_normal_sum = -0.5 * (((act_resample - act_mu) / (tf.exp(act_logstd) + epsilon)) ** 2 + 2 * act_logstd + np.log(2 * np.pi))
    # 做求和降维操作  act_resample = [batch_size, 1]
    return tf.reduce_mean(log_normal_sum, axis=1)

# SAC的公式26-27的欧拉变换：修正原始动作分布- 对重新参数化的动作分布概率计算
def euler_transformation(logp_act_resample, act_resample):
    # 再次求和降维了     =  logp_act_resample - 2 * (log 2 - x - log(exp( -2*x ) + 1))
    logp_act_resample -= tf.reduce_mean(2 * (np.log(2) - act_resample - tf.nn.softplus(-2 * act_resample)), axis=1)
    return logp_act_resample

# alpha网络更新
def masac_alpha_log(make_obs_ph_n,
                    act_space_n,
                    p_index,
                    p_func,
                    q_func,
                    optimizer,
                    grad_norm_clipping=None,
                    local_q_func=False,
                    deterministic=False,  # 固定alpha
                    target_entropy=1,  # 目标熵
                    init_alpha_log=np.log(0.2),  # 初始化alpha_log
                    num_units=64,
                    scope="trainer",
                    reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if deterministic:
            alpha_log = init_alpha_log
            np_alpha_log = np.array([init_alpha_log])
            return np_alpha_log, None
        else:
            """① 设置占位符"""
            target_entropy *= np.log(act_space_n[p_index].shape)  # agent对应的动作维度？还是 所有agent的动作维度？
            alpha_log = tf.Variable(init_alpha_log,name="alpha",dtype=tf.float32)
            alpha = tf.convert_to_tensor(tf.exp(alpha_log))
            act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
            obs_ph_n = make_obs_ph_n
            """② 设置actor网络的输入p_input"""
            p_input = obs_ph_n[p_index]
            """③ 构建actor网络-mlp"""
            act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in
                        range(len(act_space_n))]
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,
                       activation_fn=None)
            """由未归一化的[mu,logstd] 得到对应的对数高斯分布class"""
            act_pd = act_pdtype_n[p_index].pdfromflat(p)
            actpd_mu = act_pd.mean
            actpd_logstd = act_pd.logstd
            """在对数高斯分布中重新参数化，再采样"""
            act_resample = act_pd.reparameterization()  # 未归一化
            # 压缩高斯分布
            act_resample = tf.tanh(act_resample)
            # 获取重新参数化的动作的概率密度
            logp_act_resample = log_gaussian_policy(act_resample, actpd_mu, actpd_logstd)
            """原始动作分布概率密度logp_act_resample"""
            # 获取对数形式的原始动作分布
            logp_act_resample = euler_transformation(logp_act_resample, act_resample) # 欧拉变换
            alpha_losses = -(alpha * tf.stop_gradient(logp_act_resample - target_entropy))
            alpha_loss = tf.nn.compute_average_loss(alpha_losses)
            optimize_expr = U.minimize_and_clip(optimizer, alpha_loss, alpha_log, grad_norm_clipping)

            train = U.function(inputs=obs_ph_n + act_ph_n, outputs=alpha_loss, updates=[optimize_expr])
            alpha_values = U.function(inputs=obs_ph_n + act_ph_n, outputs=alpha_log)
            return alpha, train

# masac的策略网络-新增 alpha
def masac_p_train(make_obs_ph_n,
                  act_space_n,
                  p_index,
                  p_func,
                  q_func,
                  optimizer,
                  grad_norm_clipping=None,
                  fix=False,
                  local_q_func=False,
                  num_units=64,
                  scope="trainer",
                  reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        """① 设置占位符"""
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        """② 设置actor网络的输入p_input"""
        p_input = obs_ph_n[p_index]
        if fix:  # 固定α
            alpha = tf.placeholder(tf.float32, [None],name="alpha")
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), reuse=False, scope="p_func",
                       num_units=num_units,
                       activation_fn=None)
        else: # 不固定
            alpha = tf.get_variable("alpha",shape=[None,],dtype=tf.float32)
            p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), reuse=True, scope="p_func",
                       num_units=num_units,
                       activation_fn=None)
        """③ 构建actor网络-mlp"""
        """由未归一化的[mu,logstd] 得到对应的对数高斯分布class"""

        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        actpd_mu = act_pd.mean
        actpd_logstd = act_pd.logstd  # actor网络输出的logstd经(-20,2)裁剪后
        """在对数高斯分布中重新参数化，再采样"""
        # 连续：act_sample = mu + std*tf.random_normal(tf.shape(mu)) 重采样
        act_resample = act_pd.reparameterization()
        # 压缩高斯分布
        act_resample = tf.tanh(act_resample)
        # 获取重新参数化的动作的概率密度
        logp_act_resample = log_gaussian_policy(act_resample, actpd_mu, actpd_logstd)
        """获得原始动作分布概率密度logp_act_resample"""
        # 对数形式的原始动作分布——欧拉变换
        logp_act_resample = euler_transformation(logp_act_resample, act_resample)
        """④ 定义Q网络输入量类型"""
        act_input_n = act_ph_n + []
        act_input_n[p_index] = act_resample
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        """⑤ 构建critic网络-mlp"""
        q1 = q_func(q_input, 1, scope="q1_func", reuse=True, num_units=num_units)[:, 0]
        q2 = q_func(q_input, 1, scope="q2_func", reuse=True, num_units=num_units)[:, 0]
        """⑥ 计算actor的loss = （Q(s,a;θ)）^2"""
        q = tf.minimum(q1, q2)
        pg_loss = -tf.reduce_mean(q - alpha * logp_act_resample)
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        loss = pg_loss + p_reg * 1e-3
        """⑦ actor网络更新"""
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)
        """⑧ 定义function方程"""
        train = U.function(inputs=obs_ph_n + act_ph_n + [alpha], outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_resample)
        logp_act = U.function(inputs=[obs_ph_n[p_index]], outputs=logp_act_resample)
        p_values = U.function([obs_ph_n[p_index]], p)
        """⑨ （可选）构建actor_target网络"""
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units, activation_fn=None)
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        """⑩ 更新actor_targrt网络 """
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        """⑾ 目标网络的动作"""
        target_actpd_mu = act_pdtype_n[p_index].pdfromflat(target_p).mean
        target_actpd_logstd = act_pdtype_n[p_index].pdfromflat(target_p).logstd
        target_act_resample = act_pdtype_n[p_index].pdfromflat(target_p).reparameterization()
        target_logp_act_resample = log_gaussian_policy(target_act_resample, target_actpd_mu,target_actpd_logstd)
        target_logp_act_resample = euler_transformation(target_logp_act_resample, target_act_resample)
        target_act_mu = tf.tanh(target_actpd_mu)
        target_act_resample = tf.tanh(target_act_resample)
        # 定义feed_dict
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_resample)
        target_logp_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_logp_act_resample)
        target_p = U.function([obs_ph_n[p_index]], target_p)
        return act, logp_act, train, update_target_p, \
               {'target_act': target_act, 'target_logp_act': target_logp_act},


def masac_q_train(make_obs_ph_n,
                  act_space_n,
                  q_index,
                  q_func,
                  optimizer,
                  grad_norm_clipping=None,
                  local_q_func=False,
                  scope="trainer",
                  reuse=None,
                  num_units=64):
    with tf.variable_scope(scope, reuse=reuse):
        """① 设置ph占位符"""
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action" + str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        """②构建输入"""
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        """③ 构建critic网络的mlp-1层输入2层隐藏1层输出"""
        """设q1，q2网络"""
        # q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        q1 = q_func(q_input, 1, scope="q1_func", num_units=num_units)[:, 0]
        q2 = q_func(q_input, 1, scope="q2_func", num_units=num_units)[:, 0]

        q1_func_vars = U.scope_vars(U.absolute_scope_name("q1_func"))
        q2_func_vars = U.scope_vars(U.absolute_scope_name("q2_func"))
        """分别计算Q1,Q2的loss"""
        min_q = tf.minimum(q1, q2)
        q1_loss = tf.reduce_mean(tf.square(q1 - target_ph))
        q2_loss = tf.reduce_mean(tf.square(q2 - target_ph))
        # viscosity solution to Bellman differential equation in place of an initial condition
        q1_reg = tf.reduce_mean(tf.square(q1))
        q2_reg = tf.reduce_mean(tf.square(q2))
        loss1 = q1_loss + 1e-3 * q1_reg
        loss2 = q2_loss + 1e-3 * q2_reg
        """q1,q2网络的更新"""
        optimize1_expr = U.minimize_and_clip(optimizer, loss1, q1_func_vars, grad_norm_clipping)
        optimize2_expr = U.minimize_and_clip(optimizer, loss2, q2_func_vars, grad_norm_clipping)
        train1 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss1, updates=[optimize1_expr])
        train2 = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss2, updates=[optimize2_expr])
        """取min_Q值作为actor网络更新输入量"""
        min_q_values = U.function(obs_ph_n + act_ph_n, min_q)
        """④构建Q_target的神经网络mlp"""
        target_q1 = q_func(q_input, 1, scope="target_q1_func", num_units=num_units)[:, 0]
        target_q2 = q_func(q_input, 1, scope="target_q2_func", num_units=num_units)[:, 0]
        """Q_target的网络更新"""
        target_q1_func_vars = U.scope_vars(U.absolute_scope_name("target_q1_func"))
        target_q2_func_vars = U.scope_vars(U.absolute_scope_name("target_q2_func"))

        update_target_q1 = make_update_exp(q1_func_vars, target_q1_func_vars)
        update_target_q2 = make_update_exp(q2_func_vars, target_q2_func_vars)
        """⑤计算min_Q_target网络的Q_target(s',a';θ_target)"""
        min_target_q = tf.minimum(target_q1, target_q2)
        min_target_q_values = U.function(obs_ph_n + act_ph_n, min_target_q)
        return [train1, train2], [update_target_q1, update_target_q2], {'min_q_values': min_q_values, 'min_target_q_values': min_target_q_values},

class MASACAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func=False,
                 target_entorpy=1.0):  # target_entorpy目标熵（未用到）
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation" + str(i)).get())
        """① alpha自适应更新(暂缺)"""
        self.alpha_log, self.alpha_loss = masac_alpha_log(
            scope=self.name,  # agent的序号 agent%d
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr*0.03),
            grad_norm_clipping=0.5,
            deterministic=args.fix_alpha,  # 固定alpha
            target_entropy=1,
            init_alpha_log=np.log(args.init_alpha),
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        """② actor-critic网络更新"""
        self.q_train, self.q_update, self.q_debug = masac_q_train(
            scope=self.name,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 3 * 0.1),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # 用p_train建立了 策略训练网络（policy tarining network）含有（策略网络，目标策略网络）
        self.act, self.act_prob, self.p_train, self.p_update, self.p_debug, = masac_p_train(
            scope=self.name,  # agent的序号 agent%d
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr * 3 * 0.5),
            grad_norm_clipping=0.5,
            fix=args.fix_alpha,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer 创建经验缓冲区-记忆库
        self.replay_buffer = ReplayBuffer(5e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None


    def action(self, obs):
        return self.act(obs[None])[0]

    # 将经验（S,A,R,S'）存入经验记忆库replay_buffer
    def experience(self, obs, act, rew, new_obs, done, terminal):

        self.replay_buffer.add(obs, act, rew, new_obs, float(1.0)if done==True or terminal==True else float(0.0))

    def preupdate(self):
        self.replay_sample_index = None

    # 网络更新
    def update(self, agents, t):
        """① 设置更新间隔"""
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 10 == 0:
            return
        """② 提取记忆库batch"""
        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index  # 索引
        # 开始收集:
        for i in range(self.n):  # 对每个agent获取同一随机采样的数据，并合并成一个长array
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)

        num_sample = 1  # 单样本数量
        Y_target = 0.0  # Y_target
        for i in range(num_sample):
            """③ 更新actor网络"""
            if self.args.fix_alpha== True:
                alpha =np.exp(self.alpha_log)# 获取固定的alpha
            else:
                alpha = np.exp(self.alpha_log(*(obs_n+act_n)))  # 获取更新的alpha
            p_loss = self.p_train(*(obs_n + act_n+ [alpha]))
            """④ 更新critic网络"""
            # ①计算Y = r + (1-d)γ(min_q - alpha*logp_act)
            if self.args.target_actor:
                # 用目标策略网络计算联合动作a'和动作分布
                target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
                target_logp_act_next_n = agents[self.agent_index].p_debug['target_logp_act'](obs_next_n[self.agent_index])
                target_min_q_next = self.q_debug['min_target_q_values'](*(obs_next_n + target_act_next_n))
                Y_target += rew + (1 - done) * self.args.gamma * (target_min_q_next - alpha * target_logp_act_next_n)
            else:
                # 用策略网络计算a'和动作分布
                act_next_n = [agents[i].act(obs_next_n[i]) for i in range(self.n)]
                log_act_prob_next_n = agents[self.agent_index].act_prob(obs_next_n[self.agent_index])
                # 计算minQ
                target_min_q_next = self.q_debug['min_target_q_values'](*(obs_next_n + act_next_n))
                Y_target += rew + (1 - done) * self.args.gamma * (target_min_q_next - alpha * log_act_prob_next_n)  # alpha需要降维取平均
        Y_target /= num_sample
        q1_loss = self.q_train[0](*(obs_n + act_n + [Y_target]))
        q2_loss = self.q_train[1](*(obs_n + act_n + [Y_target]))
        """⑤ 更新alpha"""
        if not self.args.fix_alpha:  # 不固定
            log_alpha_loss = self.alpha_loss(*(obs_n + act_n))
        else:   # 固定
            log_alpha_loss = [None]

        self.p_update()
        for i in range(2):
            self.q_update[i]()
        return [q1_loss, q2_loss, p_loss, np.mean(Y_target), np.mean(rew), np.std(Y_target)]

    # 获取神经网络的变量
    def get_scope_var(self, scope):
        return U.scope_vars(U.absolute_scope_name(scope))
    # 保存模型
    def save_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name)
        if saver is None:
            saver = tf.train.Saver(var)
        saver.save(U.get_session(), file_name + self.name + '/')

    # 加载模型
    def load_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        var = self.get_scope_var(scope=self.name + '/')
        if saver is None:
            saver = tf.train.Saver(var)
        saver.restore(U.get_session(), file_name + self.name + '/')

    # 保存部分模型-如只保存Actor网络，且不需要目标网络
    def saver_actor_model(self, file_name, steps, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        with tf.variable_scope(self.name, reuse=None):
            var = self.get_scope_var(scope=self.actor_vars)
            if saver is None:
                saver = tf.train.Saver(var)
            saver.save(U.get_session(), file_name + self.name + '/actor/', global_step=steps)

    # 加载actor模型
    def load_actor_model(self, file_name, saver=None):
        os.makedirs(os.path.dirname(file_name + self.name), exist_ok=True)
        if saver is None:
            saver = tf.train.Saver()
        saver.restore(U.get_session(), file_name + self.name + '/actor/')






























