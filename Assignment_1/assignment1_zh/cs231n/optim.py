import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:
本文件实现了训练神经网络时常用的一阶更新规则。每个更新规则接收当前权重以及损失关于这些权重的梯度，
并产生下一组权重。每个更新规则都有相同的接口：

def update(w, dw, config=None):

Inputs:
输入：
  - w: A numpy array giving the current weights.
  - w: 给出当前权重的 numpy 数组。
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - dw: 形状与 w 相同的 numpy 数组，给出损失关于 w 的梯度。
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.
  - config: 一个字典，包含学习率、动量等超参数值。如果更新规则需要在多次迭代中缓存数值，
    那么 config 也会保存这些缓存值。

Returns:
返回：
  - next_w: The next point after the update.
  - next_w: 更新之后的下一个点。
  - config: The config dictionary to be passed to the next iteration of the
    update rule.
  - config: 要传给下一次更新规则调用的 config 字典。

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.
注意：对大多数更新规则而言，默认学习率的效果可能并不好；
不过其他超参数的默认值在各类不同问题上应该都能工作得不错。

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
为了效率，更新规则可能会执行就地更新，即修改 w 并让 next_w 等于 w。
"""


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    执行普通的随机梯度下降。

    config format:
    config 格式：
    - learning_rate: Scalar learning rate.
    - learning_rate: 标量学习率。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    执行带动量的随机梯度下降。

    config format:
    config 格式：
    - learning_rate: Scalar learning rate.
    - learning_rate: 标量学习率。
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - momentum: 0 到 1 之间的标量，表示动量值。
      令 momentum = 0 就退化为 sgd。
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    - velocity: 形状与 w、dw 相同的 numpy 数组，用来保存梯度的滑动平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    # TODO:                                                                   #
    # 实现动量更新公式。把更新后的值存到 next_w 变量中。你还应当使用并更新速度 v。 #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    使用 RMSProp 更新规则，它利用梯度平方值的滑动平均来为每个参数设置自适应的学习率。

    config format:
    config 格式：
    - learning_rate: Scalar learning rate.
    - learning_rate: 标量学习率。
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - decay_rate: 0 到 1 之间的标量，表示梯度平方缓存的衰减率。
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - epsilon: 用于平滑、避免除以零的小标量。
    - cache: Moving average of second moments of gradients.
    - cache: 梯度二阶矩的滑动平均。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    # TODO:                                                                   #
    # 实现 RMSprop 更新公式，把 w 的下一个值存到 next_w 变量中。              #
    # 不要忘记更新存在 config['cache'] 中的缓存值。                           #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    使用 Adam 更新规则，它同时利用了梯度及其平方的滑动平均，还有一个偏差校正项。

    config format:
    config 格式：
    - learning_rate: Scalar learning rate.
    - learning_rate: 标量学习率。
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta1: 梯度一阶矩滑动平均的衰减率。
    - beta2: Decay rate for moving average of second moment of gradient.
    - beta2: 梯度二阶矩滑动平均的衰减率。
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - epsilon: 用于平滑、避免除以零的小标量。
    - m: Moving average of gradient.
    - m: 梯度的滑动平均。
    - v: Moving average of squared gradient.
    - v: 梯度平方的滑动平均。
    - t: Iteration number.
    - t: 迭代次数。
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    next_w = None
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in #
    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #
    # TODO:                                                                   #
    # 实现 Adam 更新公式，把 w 的下一个值存到 next_w 变量中。                 #
    # 不要忘记更新存在 config 中的 m、v 和 t 变量。                           #
    #                                                                         #
    # 注意：为了与参考输出一致，请在把 t 用于任何计算之前先修改 t。           #
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config
