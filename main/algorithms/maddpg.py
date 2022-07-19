"""
core code for the MADDPG algorithm
（MADDPG算法的核心代码）
"""
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tools.tf_util as U

from tools.distributions import make_pdtype
from algorithms import AgentTrainer
from algorithms.replay_buffer import ReplayBuffer

# Actor-critic
def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None,activation_fn=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=activation_fn)
        return out

# r = 0
# G= for i = 1 range n:
#   r = gamma * r + reward
def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    # [::-1] 是
    return discounted[::-1]


#  参数更新
def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2  # 1.0 - 0.01 = 0.99
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def p_train(make_obs_ph_n,
            act_space_n,
            p_index,
            p_func,
            q_func,
            optimizer,
            grad_norm_clipping=None,
            local_q_func=False,
            num_units=64,
            scope="trainer",
            reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        # 创建动作概率分布--连续动作make_pdtype(act_space)为动作参数化分布
        # 根据输入的动作类型act_space_n，建立对应的概率分布函数
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders
        # 设置张量
        obs_ph_n = make_obs_ph_n  # 观测状态维度的placeholder
        # 动作空间的概率分布生成placeholder
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # 策略空间的输入变量-观测状态O或O‘
        p_input = obs_ph_n[p_index]
        # 策略网络的处理函数p_func-以model作为处理
        # 处理函数选择了神经网络NN-64个神经元-2层神经网络
        # param_shape-返回了动作范围大小
        # 生成策略网络-p是在所有动作可选范围内的输出概率
        # p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units)
        # 输出层加入tanh激活函数
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="p_func", num_units=num_units,activation_fn=None)
        # 将神经网络"p_func"的输出量，通过scope_vars转化为训练的变量列表形式（list）并附加是否可训练标志
        p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

        # wrap parameters in distribution p分布参数
        # 动作分布类型结合策略网络输出的动作p-tensor，得到动作分布act_pd
        # act_pd为 神经网络全连接层的输出结果-logit[N,1] 未softmax的概率-p个分类
        act_pd = act_pdtype_n[p_index].pdfromflat(p)
        # 在动作分布act_pd上采样输出加入了噪声N的softmax最终概率到act_sample=[a1,a2,a3,a4,a5]
        act_sample = act_pd.sample()  # 最终动作概率
        act_sample = tf.tanh(act_sample)
        # reduce_mean 计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
        # tf.square(a) 是对a里的每一个元素求平方
        # flatparam 是返回输入的tensor类型
        # 将动作概率分布的每一个值取2次方（平方），再对所有的值求和取平均值，输出到p_reg
        # p_reg为动作参数梯度-actor
        # 计算act_pd的 期望 E[act_pd]
        # act_pd.flatparam()  == p
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
        # 创建动作输入变量的类型空间
        act_input_n = act_ph_n + []
        # 对动作概率分布采样输出到 act_input_n[p_index]
        act_input_n[p_index] = act_pd.sample()
        # critic-Q网络输入变量的类型空间为：
        # 状态O+动作输入变量的类型空间placeholder进行第1个维度的张量(placeholder)拼接
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        # 判断是否使用DDPG算法
        if local_q_func:  # = True
            # critic-Q值网络输入更新为；将状态O张量空间和动作A张量空间第p_index维度的第1个子维度拼接
            # DDPG是单智能体算法
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        # 用Q值网络，输入为O,A的拼接张量空间，输出为Q值-critic
        q = q_func(q_input, 1, scope="q_func", reuse=True, num_units=num_units)[:, 0]
        # 计算Q值，降维取负平均值传递给pg_loss（策略梯度损失）
        pg_loss = -tf.reduce_mean(q)
        # 损失函数loss= 平均Q值pg_loss + 可选动作的方差p_reg* 1e-3
        loss = pg_loss + p_reg * 1e-3
        # 优化工具去最小化损失loss,优化损失-更新策略网络的变量p_func_vars == optimize_expr
        optimize_expr = U.minimize_and_clip(optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions
        # train=更新后的损失loss： 输入状态空间O+动作空间A, 输出为loss， 更新目标为更新后的Actor网络变量optimize_expr
        train = U.function(inputs=obs_ph_n + act_ph_n, outputs=loss, updates=[optimize_expr])
        # act为输出动作具体值-训练函数

        act = U.function(inputs=[obs_ph_n[p_index]], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network-> theta_target
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]), scope="target_p_func",
                          num_units=num_units,activation_fn=None)
        # 将神经网络"target_p_func"的输出量，通过scope_vars转化为训练的变量列表形式（list）并附加是否可训练标志
        target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))
        # 更新策略目标网络的参数theta_target <-- alpha_target * theta + (1 -  alpha_target) * theta_target
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
        # 目标策略网络的动作概率值--> （基于动作概率分布函数结合策略网络参数target_p)的采样
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        # 输入为 agent[p_index]的状态O’_index，输出为目标策略网络的取样动作 target_act_sample
        # 将经过函数计算的目标策略动作概率值A' ==> target_act
        target_act = U.function(inputs=[obs_ph_n[p_index]], outputs=target_act_sample)
        """
        策略网络总输出为：
                act: 策略网络输出的采样动作
                tarin: 策略网络生成的损失函数loss
                update_target_p: 目标策略网络的更新参数-软更新
                字典：p_values 策略网络中，用acotor网络的动作生成的动作Ai 
                     target_act 目标策略网络的输出-Oi‘观测状态下选择的确定性动作Ai'
        """
        return act, train, update_target_p, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n,  # 所有agent的观测状态
            act_space_n,  # 所有agent的动作
            q_index,   # agent[q_index]
            q_func,  # 用于神经网络NN的模型model = q_func
            optimizer,  # 优化算法(lr) alpha_w, alpha_w_target
            grad_norm_clipping=None,  # 梯度修剪
            local_q_func=False,  # DDPG标志
            scope="trainer",  # MADDPG名字
            reuse=None,  # reuse: Boolean型, 是否重复使用参数.
            num_units=64):
    # 变量作用域；tensorboard画流程图进行可视化封装变量
    # 在模型中开辟各自的空间，而其中的变量均在这个空间内进行管理
    # 即在模型中，所有变量都加入了前缀 scope=trainer/***
    with tf.variable_scope(scope, reuse=reuse):
        # create distribtuions
        # 为每一个智能体(agent[N])建立动作概率分布
        #print(act_space_n)
        #print('拆分')
        #for act_space in act_space_n:
        #    print(act_space)
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders  设置张量placeholder
        # 所有agent观测状态O的空间X=[O1, O2, ..., ON]
        obs_ph_n = make_obs_ph_n
        # 对每一个智能体agent[i]，根据动作概率分布，取样动作值设置张量placeholder并命名[action1, action2, ..., actionN]
        act_ph_n = [act_pdtype_n[i].sample_placeholder([None], name="action"+str(i)) for i in range(len(act_space_n))]
        # 目标(回报)价值空间-张量
        target_ph = tf.placeholder(tf.float32, [None], name="target")
        # 取 所有agent的状态空间X 与 动作空间A 进行拼接作为Q值网络的输入量q_input
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        # 若是DDPG
        if local_q_func:
            # 更改Q值网络的输入量q_input的计算方式：
            # 取agent[index]的观测状态O和动作空间A的q_index的内部子集拼接
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        # 输入：O与A的拼接量q_input, 1为输出层数， 神经网络命名q_func,隐藏层神经元数64，全连接层
        # 输出为 第0列的列向量的q值-Q(O,A;w)
        q = q_func(q_input, 1, scope="q_func", num_units=num_units)[:, 0]
        # scope_vars：对q_func神经网络的作用域deepq/q_func筛选
        #             满足GLOBAL_VARIABLES的变量，即用tf.Variable()和tf.get_variable()创建的变量
        #             转化为训练的数据（列表形式），默认False 不加标志位
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
        # 计算价值损失L(W)--> average[(q - target_ph)^2]
        q_loss = tf.reduce_mean(tf.square(q - target_ph))
        # viscosity solution to Bellman differential equation in place of an initial condition
        # ave(q^2) ~ Bellman微分方程的粘度解 代替 初始条件
        # 即 对agent[index]，降维->o下所有动作q的均方值ave(q^2) 作为 Q(o,a)的近似解
        q_reg = tf.reduce_mean(tf.square(q))
        # 修正真正的动作价值损失 loss = 动作价值损失q_loss + 0.001 * bellman微分方程近似解q_reg
        #积分方程： loss = 1e-3 *q_reg (t=0初始条件) + q_loss
        loss = q_loss + 1e-3 * q_reg
        # 用优化器optimizer-学习率lr-alpha_w, 输入量为loss， 训练数据为q_func_vars，梯度修剪grad_norm_clipping
        # 对loss关于q_func_vars求梯度(导数)，optimize_expr为更新后的q_func_vars-Q值神经网络参数w
        optimize_expr = U.minimize_and_clip(optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions-创建可调用函数
        # 用function， 输入为 观测状态O+动作A+回报U（target_q_values）， 输出为损失loss， 更新目标为 梯度后的Q值神经网络参数w
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph], outputs=loss, updates=[optimize_expr])
        # Q值的计算结果： 输入为观测状态O+动作A，以及对应的价值网络输出的Q值
        # 最终动作价值为q_values
        q_values = U.function(obs_ph_n + act_ph_n, q)
        """
        target network
        """
        # 目标价值网络-critic_target  参数：w_target
        # 输入为q_input-观测状态X+动作A，输出为1阶数，神经网络名称:target_q_fuc, 隐藏层64个神经元，全连接层MLP
        # 输出为目标价值网络Q‘值-每个动作的Q价值-为列向量
        target_q = q_func(q_input, 1, scope="target_q_func", num_units=num_units)[:, 0]
        # 将目标价值网络的参数转换为训练的变量-列表形式（list）并附加可训练标志
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
        # 更新 目标价值网阔的参数w_target <-- alpha_target * w（q_func_vars） + (1 - alpha_target) * w_target（target_q_func_vars）
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars)
        # 根据 所有agent的观测状态O+动作A以及对应的Q‘值Q'(O',A';w_target),计算得到目标价值网络输出结果-Q'(O',A';w_target)
        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)
        """
        critic价值网络输出结果为：
            train: 价值网络计算得到的损失函数loss--列表形式
            update_target_q: 目标价值网络的参数更新式
            字典：
                q_valus: 价值网络输出的最终Q值-Q(O,A;w)
                target_q_values: 目标价值网络输出的最终Q'值-Q(O',A';w_target)
        """
        return train, update_target_q, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGAgentTrainer(AgentTrainer):
    """
    主体MADDPGAgentTrainer函数介绍：
        1-init
            初始化agent的参数--name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func
            初始化价值网络--q_train，q_update，q_debug；初始化策略网络--act，p_train，p_update，p_debug
            初始化经验缓存区-记忆库--replay_buffer
        2-action
            根据观测状态O，选择动作————根据策略网络alpha
            策略神经网络- 输入为状态O,输出为动作A
        3-experience
            # 将经验（S,A,R,S'）存入经验记忆库replay_buffer
        4-preupdate
            重置记忆库取样索引
        5-update
            在更新的时候下计算每一个样本的目标Q值-Q'(O',A';w_target),输入q_train,p_train神经网络训练
            最后更新整个神经网络
    """
    # 类初始化：
    # 智能体所用trainer(训练者)的name,
    # 神经网络model,
    # 状态空间维度obs_shape_n,
    # 动作空间维度act_space_n,
    # agent的索引agent_index
    # args 是超参数
    # local_q_func=True ==> DDPG算法
    def __init__(self, name, obs_shape_n, act_space_n, agent_index, args, local_q_func=False):
        self.name = name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(obs_shape_n[i], name="observation"+str(i)).get())

        # Create all the functions necessary to train the model
        # 创建训练模型所需的所有函数
        # 用q_train建立了 价值训练网络（value training network）含有（价值网络，目标价值网络）

        self.q_train, self.q_update, self.q_debug = q_train(
            scope=self.name,  # agent的名字， agent1,agent2,...,agentN
            make_obs_ph_n=obs_ph_n,  # agent的状态空间--tensor（所有agent的观测状态空间）
            act_space_n=act_space_n,  # agent的动作空间--list（所有agent的动作空间）
            q_index=agent_index,  # 第index个agent
            # model为网络-在Q价值网络中为critic
            q_func=mlp_model,  # 状态O'下根据actor目标网络参数theta_target输出的动作A'，计算得到的Q-真实值
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr*0.1),  # 0.001优化器-价值网络的学习率args.lr
            grad_norm_clipping=0.5,  # 梯度修剪
            local_q_func=local_q_func,  # = True 为DDPG算法
            num_units=args.num_units # arg貌似是所有超参数的类 args.num_units神经网络的64个神经元
        )
        self.act, self.p_train, self.p_update, self.p_debug = p_train(
            scope=self.name,  # agent的序号 agent%d
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=mlp_model,
            q_func=mlp_model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units
        )
        # Create experience buffer
        self.replay_buffer = ReplayBuffer(5e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None


    def action(self, obs):
        return self.act(obs[None])[0]

    # 将经验（S,A,R,S'）存入经验记忆库replay_buffer
    def experience(self, obs, act, rew, new_obs, done, terminal):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def preupdate(self):
        self.replay_sample_index = None

    # 更新
    def update(self, agents, t):
        # 更新的条件:
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % 100 == 0:
            return

        self.replay_sample_index = self.replay_buffer.make_index(self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        # 开始收集:
        for i in range(self.n):
            obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        # 获取self.agent的buffer
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)
        # train q network
        num_sample = 1
        target_q = 0.0
        for i in range(num_sample):  # 采集样本训练
            target_act_next_n = [agents[i].p_debug['target_act'](obs_next_n[i]) for i in range(self.n)]
            target_q_next = self.q_debug['target_q_values'](*(obs_next_n + target_act_next_n))
            # Q = R + gamma * (1 - done——终止条件) * Q
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next  # [batch_size, 1]
        target_q /= num_sample
        q_loss = self.q_train(*(obs_n + act_n + [target_q]))
        # train p network
        p_loss = self.p_train(*(obs_n + act_n))
        self.p_update()
        self.q_update()
        return [q_loss, p_loss, np.mean(target_q), np.mean(rew), np.std(target_q)]

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




