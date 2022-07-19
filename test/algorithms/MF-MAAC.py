
"""
core code for the MF-MAAC algorithm
（大规模的平均场MARL算法的核心代码）
"""
import numpy as np
import random
import tensorflow as tf
import maddpg.common.tf_util as U
from maddpg.common.distributions import make_pdtype
from maddpg import AgentTrainer
from maddpg.trainer.replay_buffer import ReplayBuffer
