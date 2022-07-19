import tensorflow as tf
import numpy as np
import tools.tf_util as U
from tensorflow.python.ops import math_ops
from .multi_discrete import MultiDiscrete
from tensorflow.python.ops import nn

class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError
    def mode(self):
        raise NotImplementedError
    def logp(self, x):
        raise NotImplementedError
    def kl(self, other):
        raise NotImplementedError
    def entropy(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError

class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError
    def pdfromflat(self, flat):
        return self.pdclass()(flat)
    def param_shape(self):
        raise NotImplementedError
    def sample_shape(self):
        raise NotImplementedError
    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class CategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return CategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return []
    def sample_dtype(self):
        return tf.int32

class SoftCategoricalPdType(PdType):
    def __init__(self, ncat):
        self.ncat = ncat
    def pdclass(self):
        return SoftCategoricalPd
    def param_shape(self):
        return [self.ncat]
    def sample_shape(self):
        return [self.ncat]
    def sample_dtype(self):
        return tf.float32

class MultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.ncats = high - low + 1
    def pdclass(self):
        return MultiCategoricalPd
    def pdfromflat(self, flat):
        return MultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        # []
        return [sum(self.ncats)]
    def sample_shape(self):
        return [len(self.ncats)]
    def sample_dtype(self):
        return tf.int32

class SoftMultiCategoricalPdType(PdType):
    def __init__(self, low, high):
        self.low = low    # [0,0,0,0]
        self.high = high  # [n-1,n-1,n-1,n-1]
        self.ncats = high - low + 1  # 维度数
    def pdclass(self):
        return SoftMultiCategoricalPd
    def pdfromflat(self, flat):
        # flat = p
        return SoftMultiCategoricalPd(self.low, self.high, flat)
    def param_shape(self):
        return [sum(self.ncats)]
    def sample_shape(self):
        return [sum(self.ncats)]
    def sample_dtype(self):
        return tf.float32

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return DiagGaussianPd
    def param_shape(self):
        return [2*self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.float32

class BernoulliPdType(PdType):
    def __init__(self, size):
        self.size = size
    def pdclass(self):
        return BernoulliPd
    def param_shape(self):
        return [self.size]
    def sample_shape(self):
        return [self.size]
    def sample_dtype(self):
        return tf.int32

# WRONG SECOND DERIVATIVES
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.nn.softmax(logits)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return U.argmax(self.logits, axis=1)
#     def logp(self, x):
#         return -tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, x)
#     def kl(self, other):
#         return tf.nn.softmax_cross_entropy_with_logits(other.logits, self.ps) \
#                 - tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def entropy(self):
#         return tf.nn.softmax_cross_entropy_with_logits(self.logits, self.ps)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)

class CategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.argmax(self.logits, axis=1)
    def logp(self, x):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return U.argmax(self.logits - tf.log(-tf.log(u)), axis=1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftCategoricalPd(Pd):
    def __init__(self, logits):
        self.logits = logits
    def flatparam(self):
        return self.logits
    def mode(self):
        return U.softmax(self.logits, axis=-1)
    def logp(self, x):
        return -tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
    def kl(self, other):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        a1 = other.logits - U.max(other.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        z1 = U.sum(ea1, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=1)
    def entropy(self):
        a0 = self.logits - U.max(self.logits, axis=1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = U.sum(ea0, axis=1, keepdims=True)
        p0 = ea0 / z0
        return U.sum(p0 * (tf.log(z0) - a0), axis=1)
    def sample(self):
        """这是传说中的OU-noise噪声？  不是，简单的噪声"""
        # tf.shape 将矩阵的维度输出为一个维度矩阵，返回仍是张量，代表形状参数
        # tf.random_uniform 0-1均匀分布中获取随机值，返回用于填充随机均匀值的指定形状的张量
        # logits 是没有下界和上界的数
        u = tf.random_uniform(tf.shape(self.logits))  # 个人认为是生成了噪声
        # tf.log 计算TensorFlow的自然对数
        # tf.log(-tf.log(u)) 使得原本范围为0~1的u值，变得更大了。
        # 引入噪声，进行分类softmax生成概率
        return U.softmax(self.logits - tf.log(-tf.log(u)), axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)        

class MultiCategoricalPd(Pd):
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.int32)
        self.categoricals = list(map(CategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        return self.low + tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
    def logp(self, x):
        return tf.add_n([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        return self.low + tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class SoftMultiCategoricalPd(Pd):  # doesn't work yet
    def __init__(self, low, high, flat):
        self.flat = flat
        self.low = tf.constant(low, dtype=tf.float32)
        self.categoricals = list(map(SoftCategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
    def flatparam(self):
        return self.flat
    def mode(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].mode())
        return tf.concat(x, axis=-1)
    def logp(self, x):
        return tf.add_n([p.logp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
    def kl(self, other):
        return tf.add_n([
                p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
            ])
    def entropy(self):
        return tf.add_n([p.entropy() for p in self.categoricals])
    def sample(self):
        x = []
        for i in range(len(self.categoricals)):
            x.append(self.low[i] + self.categoricals[i].sample())
        return tf.concat(x, axis=-1)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat  # flat = p = [mu ,logsted]
        mean, logstd = tf.split(axis=1, num_or_size_splits=2, value=flat)
        logstd = tf.clip_by_value(logstd,-20,2)  # 剪切-经验操作(-20,2)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)
    def flatparam(self):  # 返回的是 神经网络的原始输出值
        return self.flat        

    def logp(self, x):
        return - 0.5 * U.sum(tf.square((x - self.mean) / self.std), axis=1) \
               - 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[1]) \
               - U.sum(self.logstd, axis=1)
    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return U.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=1)
    def entropy(self):
        #  log var + 0.5* log(2*2π*e)
        return U.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), 1)
    # 采样sample  # 从固定的均值，方差中采样
    def sample(self):
        # return tf.tanh(tf.random_normal(tf.shape(self.mean),mean=self.mean,stddev=self.std))
        return tf.tanh(self.mean + self.std * tf.random_normal(tf.shape(self.mean)))
    def evaluation_sample(self):
        return tf.tanh(self.mean + self.std)
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)
    """**********************SAC的重新参数化系列操作*********************************"""
    # 对数形式高斯分布
    def log_gaussian_policy(self, act_resample, act_mu, act_logstd):
        MIN_LIM = 1e-8
        log_normal_sum = -0.5 * (((act_resample - act_mu) / (tf.exp(log_std) + MIN_LIM)) ** 2 + 2 * act_logstd + np.log(
            2 * np.pi))
        return tf.reduce_mean(log_normal_sum, axis=1)
    # SAC的公式26-27的欧拉变换：
    def euler_transformation(self, lpgp_act_resample, act_resample):
        lpgp_act_resample -= tf.reduce_mean(2 * (np.log(2) - pi - tf.nn.softplus(-2 * pi)), axis=1)
        return lpgp_act_resample
    # 重新参数化-可以看做是 从标准高斯分布中重新采样
    def reparameterization(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
    def evaluation_reparameterization(self):
        return self.mean + self.std

class BernoulliPd(Pd):
    def __init__(self, logits):
        self.logits = logits
        self.ps = tf.sigmoid(logits)
    def flatparam(self):
        return self.logits
    def mode(self):
        return tf.round(self.ps)
    def logp(self, x):
        return - U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=1)
    def kl(self, other):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=1) - U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def entropy(self):
        return U.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=1)
    def sample(self):
        p = tf.sigmoid(self.logits)
        u = tf.random_uniform(tf.shape(p))
        return tf.to_float(math_ops.less(u, p))
    @classmethod
    def fromflat(cls, flat):
        return cls(flat)

def make_pdtype(ac_space):
    from gym import spaces
    # isinstance  函数来判断一个对象是否是一个已知的类型
    # 判断ac_space 是 spaces.Box 的已知类型

    # print('make_pdtype:' ,ac_space)
    #print('len(ac_space.shape)',ac_space.shape[0])
    if isinstance(ac_space, spaces.Box):  # 若 是
        assert len(ac_space.shape) == 1  # 判断 ac_space.shape长度 = 1
        # 返回 对角高斯概率分布
        return DiagGaussianPdType(ac_space.shape[0])
    elif isinstance(ac_space, spaces.Discrete):  # 若 是 spaces.Discrete
        # return CategoricalPdType(ac_space.n)
        # 软分类概率分布
        return SoftCategoricalPdType(ac_space.n)
    elif isinstance(ac_space, MultiDiscrete):  # 若 是 MultiDiscrete
        #return MultiCategoricalPdType(ac_space.low, ac_space.high)
        # 多变量软分类概率分布
        return SoftMultiCategoricalPdType(ac_space.low, ac_space.high)
    elif isinstance(ac_space, spaces.MultiBinary):  # 若 是 spaces.MultiBinary
        # 伯努利概率分布
        return BernoulliPdType(ac_space.n)
    else:  # 若全不是
        raise NotImplementedError

def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]
