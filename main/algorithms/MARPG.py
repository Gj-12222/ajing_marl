#  对RPG遗憾策略梯度进行多智能体扩展
# MARPG核心算法 on-policy-转换到off-policy
"""
主要是修改了Actor网络的目标函数
n-step TD(λ)作为Critic网络更新"""
from maddpg import AgentTrainer
import tensorflow.layers as layers
import tensorflow as tf
import maddpg.common.tf_util as U

# n-step TD(λ)
#G^λ(t) = (1 - λ)*∑n=1~T(λ^n*G^n(t))
#G^n(t) = r1 + γ*r2 + γ^2*r3 + γ^n-1*rn + γ^n*Q(sn,an)
#G^n(t) = ∑t=0~n(γ^(t-1)*rt) + γ^n*Q(sn,an)
#rewards = [1, t_size]
def n_step_td_lambda(rewards, done_masks, gamma):
    lamda = 0.8  # 超参数
    discounted = []
    r = 0
    # 逆序计算 n =T ,T-1,T-2,...,0
    for reward, done_mask in zip(rewards[::-1], done_masks[::-1]):
        r = reward + gamma * r
        r = r * (1.0 - done_mask)
        discounted.append(r)
    # 再逆序排序,即为正序：[G^λ(t)] t= 0,1,2,...,T
    return discounted[::-1]


# 普通的全连接层-和MADDPG一致  Actor-Critic共用一个
def marpg_nn(inputs,out_nums,scope,reues=False, num_units=128,activation_fn=None):
    with tf.variable_scope(scope,reues=reues):
        out = inputs
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=out_nums,  activation_fn=activation_fn)
        return out
def q_train(q_index,
            q_model,
            obs_ph_n,
            act_shape_n,
            q_optimizer,
            scope="trainer",
            grad_norm_clipping=None,
            local_q_func=False,
            reues=None,
            num_units=128):

    pass


# MARPG算法核心程序
class MARPGTrainer(AgentTrainer):
    # 初始化：①agent相关参数②Critic网络③Actor网络④记忆库
    def __init__(self,name,
                 agent_index,
                 rpg_model,
                 obs_shape_n,
                 act_shape_n,
                 args,
                 local_q_func=False):
        #①agent相关参数
        self.n= len(obs_shape_n)
        self.agent_index = agent_index
        self.name = name
        self.args = args
        #②AC网络
        obs_ph_n = []
        for i in ragne(self.n):
            # 创建BatchInput,需要get()拿到这个值
            obs_ph_n.append(U.BatchInput(obs_shape_n[i],dtype=tf.float32,name="observation"+str(i)).get()) 
        #Critic网络
        q_train(q_index=agent_index,
                scope=name,
                q_model=rpg_model,
                obs_ph_n=obs_ph_n,
                act_shape_n=act_shape_n,
                q_optimizer= tf.train.AdamOptimizer(learning_rate= args.lr * 0.1),
                num_units=args.num_units,
                grad_norm_clipping=0.5,
                local_q_func=local_q_func)





# 用off_olicy代替on_policy
class faske_on_policy_replay_buffer(object):

    def __init__(self):
        pass











