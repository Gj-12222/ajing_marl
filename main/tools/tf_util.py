import collections
import numpy as np
import os
import tensorflow as tf

def sum(x, axis=None, keepdims=False):
    return tf.reduce_sum(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def mean(x, axis=None, keepdims=False):
    return tf.reduce_mean(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def var(x, axis=None, keepdims=False):
    meanx = mean(x, axis=axis, keepdims=keepdims)
    return mean(tf.square(x - meanx), axis=axis, keepdims=keepdims)
def std(x, axis=None, keepdims=False):
    return tf.sqrt(var(x, axis=axis, keepdims=keepdims))
def max(x, axis=None, keepdims=False):
    return tf.reduce_max(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def min(x, axis=None, keepdims=False):
    return tf.reduce_min(x, axis=None if axis is None else [axis], keep_dims = keepdims)
def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)
def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)
def softmax(x, axis=None):
    return tf.nn.softmax(x, axis=axis)

# ================================================================
# Misc
# ================================================================


def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0

# ================================================================
# Inputs
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplemented()

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplemented()


class PlacholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder.
        常规tensorflow占位符的包装器。
        """
        # 常规tensorflow占位符的包装器。
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: data}


class BatchInput(PlacholderTfInput):
    def __init__(self, shape, dtype=tf.float32, name=None):
        """Creates a placeholder for a batch of tensors of a given shape and dtype
        为一批给定形状和dtype的张量创建占位符
        Parameters
        ----------
        shape: [int]
            shape of a single elemenet of the batch 批量的单个元素的形状
        dtype: tf.dtype
            number representation used for tensor contents 用于张量内容的数字表示
        name: str
            name of the underlying placeholder 基础占位符的名称
        """
        super().__init__(tf.placeholder(dtype, [None] + list(shape), name=name))


class Uint8Input(PlacholderTfInput):
    def __init__(self, shape, name=None):
        """Takes input in uint8 format which is cast to float32 and divided by 255
        before passing it to the model.

        On GPU this ensures lower data transfer times.

        Parameters
        ----------
        shape: [int]
            shape of the tensor.
        name: str
            name of the underlying placeholder
        """

        super().__init__(tf.placeholder(tf.uint8, [None] + list(shape), name=name))
        self._shape = shape
        self._output = tf.cast(super().get(), tf.float32) / 255.0

    def get(self):
        return self._output


def ensure_tf_input(thing):
    """Takes either tf.placeholder of TfInput and outputs equivalent TfInput"""
    if isinstance(thing, TfInput):
        return thing
    elif is_placeholder(thing):
        return PlacholderTfInput(thing)
    else:
        raise ValueError("Must be a placeholder or TfInput")

# ================================================================
# Mathematical utils
# ================================================================


def huber_loss(x, delta=1.0):
    """Reference: https://en.wikipedia.org/wiki/Huber_loss"""
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

# ================================================================
# Optimizer utils 优化工具-梯度求解
# ================================================================


def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in
    `var_list` while ensure the norm of the gradients for each
    variable is clipped to `clip_val`
    """
    """
    使用optimizer(优化器-优化函数)变量var_list最小化objective，
    同时确保每个变量的梯度的标准是“clip_val” 使得objective关于clip_val求导
    """
    if clip_val is None: # 不做梯度修正，直接输出梯度（元组形式）
        # 对objective求var_list的梯度(导数)，输出为元组列表[(梯度，变量)]
        return optimizer.minimize(objective, var_list=var_list)
    else:  # 做梯度修正，防止梯度消失和梯度爆炸
        # 对objective求var_list的梯度(导数)，，计算梯度gradients = (gradient, var_list)
        # Note：计算loss中可训练的var_list中的梯度。返回 (gradient, variable) 样式成对的 list
        gradients = optimizer.compute_gradients(objective, var_list=var_list)
        # i,(grad, var)==第i个值(起始为0)，(梯度，变量) 从梯度gradients中索引
        for i, (grad, var) in enumerate(gradients):  # enumerate 用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，
            if grad is not None:
                # clip_by_norm对梯度进行裁剪，通过控制梯度的最大范式，防止梯度爆炸的问题，是一种比较常用的梯度规约的方式
                gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
        # 返回： 将梯度gradients作为输入对 神经网络的变量var_list 更新后的var_list
        return optimizer.apply_gradients(gradients)
"""
apply_gradients(
    grads_and_vars,
    global_step=None,
    name=None
)
"""


# ================================================================
# Global session
# ================================================================

def get_session():
    """Returns recently made Tensorflow session"""
    # 返回最近创建的Tensorflow会话
    return tf.get_default_session()


def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    # 返回一个会话，将使用 1个CPU
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)


def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    # 返回一个只使用单个CPU的会话
    # return make_session(1)
    return make_session(16)


ALREADY_INITIALIZED = set()


def initialize():
    """Initialize all the uninitialized variables in the global scope."""
    # 对全局范围中的所有未初始化变量进行初始化
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)


# ================================================================
# Scopes--将scope变量转化为字符串-列表形式的变量
# ================================================================


def scope_vars(scope, trainable_only=False):
    """
    Get variables inside a scope
    The scope can be specified as a string
    获取作用域内的变量
    可以将作用域指定为字符串

    Parameters
    ----------
    scope: str or VariableScope
        scope in which the variables reside.
    scope :str或变量范围
        scope所在的变量位置。
    trainable_only: bool
        whether or not to return only the variables that were marked as trainable.
        是否只返回标记为可训练的变量。
    Returns
    -------
    vars: [tf.Variable]
        list of variables in `scope`.
    """
    """
    获取scope的变量
    将scope转化为字符串
    参数： scope：字符串str 或者变量所在的范围[0,~]
          trainable_only: 布尔值（0，1） 标记scope是否为可训练的字符串
    返回：为变量(神经网络形式)——将scope转化为 变量的列表形式（list）
    """
    # tf.get_collection——从一个集合中取出变量，可以找到想要的变量
    # tf.get_collection(key, scope=None)，
    # key为选择集合的标准名称， scope为筛选条件，对结果列表进行筛选，以只包含名称属性使用re.match匹配的项
    # TRAINABLE_VARIABLES ，GLOBAL_VARIABLES  是变量或训练参数的结合的标准名称
    # TRAINABLE_VARIABLES 将由优化器训练的变量对象的子集。
    # GLOBAL_VARIABLES 变量对象的默认集合，
    # 由tf.Variable()和tf.get_variable()创建的变量，会自动加入tf.GraphKeys.GLOBAL_VARIABLES中
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES,
        scope=scope if isinstance(scope, str) else scope.name
    )


def scope_name():
    """Returns the name of current scope as a string, e.g. deepq/q_func"""
    """以字符串形式返回当前作用域的名称，例如deepq/q_func"""
    return tf.get_variable_scope().name


def absolute_scope_name(relative_scope_name):
    """Appends parent scope name to `relative_scope_name`"""
    """ "将父作用域名附加到" relative_scope_name """
    return scope_name() + "/" + relative_scope_name

# ================================================================
# Saving variables-保存变量
# ================================================================


def load_state(fname, saver=None):
    """Load all the variables to the current session from the location <fname>"""
    """从位置加载所有变量到当前会话"""
    if saver is None:
        saver = tf.train.Saver()
    saver.restore(get_session(), fname)
    return saver


def save_state(fname, saver=None):
    """Save all the variables in the current session to the location <fname>"""
    """将当前会话中的所有变量保存到位置"""
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if saver is None:
        saver = tf.train.Saver()
    saver.save(get_session(), fname)
    return saver

# ================================================================
# Theano-like Function
# ================================================================

"""
I. function函数的输入输出：
    输入：将要输入数据的placeholders
    输出：将要根据placeholders进行计算的表达式, 返回Lambda函数
    （需要用到python中的lambda函数）
"""
def function(inputs, outputs, updates=None, givens=None):
    """Just like Theano function. Take a bunch of tensorflow placeholders and expersions
    computed based on those placeholders and produces f(inputs) -> outputs. Function f takes
    values to be feed to the inputs placeholders and produces the values of the experessions
    in outputs.

    Input values can be passed in the same order as inputs or can be provided as kwargs based
    on placeholder name (passed to constructor or accessible via placeholder.op.name).
    就像Theano函数一样。取一堆张量流占位符和基于这些占位符计算的表达式，并生成f(输入)->输出。函数f接受输入占位符的值，并在输出中生成表达式的值。
    输入值可以按照与输入相同的顺序传递，也可以根据占位符名称作为kwargs提供(传递给构造函数或通过placeholder.op.name访问)。
    # 功能：

    Example:
        x = tf.placeholder(tf.int32, (), name="x")
        y = tf.placeholder(tf.int32, (), name="y")
        z = 3 * x + 2 * y
        lin = function([x, y], z, givens={y: 0})

        with single_threaded_session():
            initialize()

            assert lin(2) == 6
            assert lin(x=3) == 9
            assert lin(2, 2) == 10
            assert lin(x=2, y=3) == 12

    Parameters
    ----------
    inputs: [tf.placeholder or TfInput]
        list of input arguments
    outputs: [tf.Variable] or tf.Variable
        list of outputs or a single output to be returned from function. Returned
        value will also have the same shape.
    """
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, (dict, collections.OrderedDict)):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *args, **kwargs: type(outputs)(zip(outputs.keys(), f(*args, **kwargs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *args, **kwargs: f(*args, **kwargs)[0]


# 函数功能：简化了tensorflow的f eed_dict过程
class _Function(object):
    # 初始化定义：# _Function类：
    # 初始化:
    #     检查constructor中的inputs是否是TfInput的一个子类
    #         （TfInput是PlacholderTfInput, BatchInput, Uint8Input的父类,后两者是PlacholderTfInput的子类）
    #     对类内的变量进行赋值
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        for inpt in inputs:
            if not issubclass(type(inpt), TfInput):
                assert len(inpt.op.inputs) == 0, "inputs should all be placeholders of rl_algs.common.TfInput"
        self.inputs = inputs
        updates = updates or []
        self.update_group = tf.group(*updates)
        self.outputs_update = list(outputs) + [self.update_group]
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan

    # feed_input()
    #     更新feed_dict的值，向其添加新的键值对
    def _feed_input(self, feed_dict, inpt, value):
        if issubclass(type(inpt), TfInput):
            feed_dict.update(inpt.make_feed_dict(value))
        elif is_placeholder(inpt):
            feed_dict[inpt] = value
    # __call__(*args, *kwargs)函数：（让_Function的对象能被调用）
    # 该函数的目的就是将给_Function()对象的输入，作为feed_dict传给Placeholder，最后输出激活placeholder之后的值。
    def __call__(self, *args, **kwargs):
        assert len(args) <= len(self.inputs), "Too many arguments provided"
        feed_dict = {}
        # Update the args
        for inpt, value in zip(self.inputs, args):
            self._feed_input(feed_dict, inpt, value)
        # Update the kwargs
        kwargs_passed_inpt_names = set()
        for inpt in self.inputs[len(args):]:
            inpt_name = inpt.name.split(':')[0]
            inpt_name = inpt_name.split('/')[-1]
            assert inpt_name not in kwargs_passed_inpt_names, \
                "this function has two arguments with the same name \"{}\", so kwargs cannot be used.".format(inpt_name)
            if inpt_name in kwargs:
                kwargs_passed_inpt_names.add(inpt_name)
                self._feed_input(feed_dict, inpt, kwargs.pop(inpt_name))
            else:
                assert inpt in self.givens, "Missing argument " + inpt_name
        assert len(kwargs) == 0, "Function got extra arguments " + str(list(kwargs.keys()))
        # Update feed dict with givens.
        for inpt in self.givens:
            feed_dict[inpt] = feed_dict.get(inpt, self.givens[inpt])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1]
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results



