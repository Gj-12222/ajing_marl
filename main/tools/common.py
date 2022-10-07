import numpy as np
import tensorflow as tf
import random
# seed
def seed_np_tf_random(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
