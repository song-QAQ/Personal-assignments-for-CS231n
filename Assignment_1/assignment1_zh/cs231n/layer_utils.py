from .layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
    便捷层：先做仿射变换，再接一个 ReLU

    Inputs:
    输入：
    - x: Input to the affine layer
    - x: 仿射层的输入
    - w, b: Weights for the affine layer
    - w, b: 仿射层的权重

    Returns a tuple of:
    返回一个元组：
    - out: Output from the ReLU
    - out: ReLU 的输出
    - cache: Object to give to the backward pass
    - cache: 传给反向传播的对象
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    affine-relu 便捷层的反向传播
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

