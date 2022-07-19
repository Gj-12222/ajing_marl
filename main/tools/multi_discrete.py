# An old version of OpenAI Gym's multi_discrete.py. (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

import numpy as np

import gym
# from gym.spaces import prng

"""
-多离散动作空间由一系列具有不同参数的离散动作空间组成
-它既可以适应于离散的动作空间，也可以适应于连续的(盒子)动作空间
-这是有用的代表游戏控制器或键盘，每个键可以表示为一个离散的行动空间
-它的参数化是通过传递一个数组的数组包含[min, max]为每个离散的动作空间
其中离散动作空间可以取从' min '到' max '的任何整数(包括' max ')
注意:值0总是需要表示NOOP动作。
例如:任天堂游戏控制器
-可以概念化为3个离散的动作空间:
1)方向键:离散的5 - NOOP[0]，上[1]，右[2]，下[3]，左[4]参数:min: 0, max: 4
2)按键A:离散2 - NOOP[0]，按[1]-参数:min: 0, max: 1
3)按键B:离散2 - NOOP[0]，按[1]-参数:min: 0, max: 1
—可以初始化为
多离散([[0,4]，[0,1]，[0,1]])
"""
# 继承父类gym.Space
class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    # 初始化
    def __init__(self, array_of_param_array):
        # np.array 创建数组
        self.low = np.array([x[0] for x in array_of_param_array])  # array_of_param_array的第0列
        self.high = np.array([x[1] for x in array_of_param_array])  # array_of_param_array的第1列
        self.num_discrete_space = self.low.shape[0]  # low.shape表示矩阵low的维度大小,shape[0]取矩阵行数

    # 一次取样
    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        """返回一个数组，其中包含来自每个离散动作空间的一个样本"""
        # For each row: round(random .* (max - min) + min, 0)
        # random_array = prng.np_random.rand(self.num_discrete_space)代替-伪随机数：
        random_array = np.random.rand(self.num_discrete_space)  # 产生一个矩阵low一样维度的随机数
        # 返回：[ int ( [ (high-low + 1) .* random_array ] + low ) ]
        # floor:返回数字的向下取舍整数,与int强制类型转换一致，小数位归0，整数不变
        # multiply:数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    # 包含
    def contains(self, x):
        # a.all()--保证a矩阵的所有值都满足判断，返回一个True，否则返回False
        # x的长度== low矩阵第一维度的长度(二维中为行数) 返回True，否则返回False
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property   # property 负责把一个方法变成属性调用的，把__repr__和__eq__变成属性调用
    def shape(self):
        return self.num_discrete_space  # 返回low矩阵第一维度的长度(二维中为行数)
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    # 数组self.low = other.low, 返回True
    # 数组self.high = other.high, 返回True
    #
    def __eq__(self, other):
        # 返回布尔值Bool--> True or False
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)