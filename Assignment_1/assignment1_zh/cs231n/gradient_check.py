from __future__ import print_function
from builtins import range
from past.builtins import xrange

import numpy as np
from random import randrange


def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    a naive implementation of numerical gradient of f at x
    f 在 x 处的数值梯度的朴素实现
    - f should be a function that takes a single argument
    - f 应当是一个只接受单个参数的函数
    - x is the point (numpy array) to evaluate the gradient at
    - x 是求梯度所在的点（numpy 数组）
    """

    fx = f(x)  # evaluate function value at original point
    # 在点 x 处求函数值（original point 指扰动之前的那个点，不是坐标原点）
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    # 遍历 x 的所有下标
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h
        # 在 x+h 处求函数值
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # 加上 h
        fxph = f(x)  # evalute f(x + h)
        # 计算 f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        # 计算 f(x - h)
        x[ix] = oldval  # restore
        # 恢复原值

        # compute the partial derivative with centered formula
        # 用中心差商公式计算偏导数
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        # 即斜率
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension
        # 前进到下一个维度

    return grad


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    为一个接受 numpy 数组并返回 numpy 数组的函数计算数值梯度。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """
    Compute numeric gradients for a function that operates on input
    and output blobs.
    为一个作用于输入 blob 和输出 blob 的函数计算数值梯度。

    We assume that f accepts several input blobs as arguments, followed by a
    blob where outputs will be written. For example, f might be called like:
    假定 f 接受若干个输入 blob 作为参数，后面再跟一个用于写出输出的 blob。
    例如，f 可能这样调用：

    f(x, w, out)

    where x and w are input Blobs, and the result of f will be written to out.
    其中 x 和 w 是输入 Blob，f 的结果会写到 out 中。

    Inputs:
    输入：
    - f: function
    - f: 函数
    - inputs: tuple of input blobs
    - inputs: 输入 blob 组成的元组
    - output: output blob
    - output: 输出 blob
    - h: step size
    - h: 步长
    """
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=["multi_index"], op_flags=["readwrite"])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(
        lambda *args: net.forward(), inputs, output, h=h
    )


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    sample a few random elements and only return numerical
    in this dimensions.
    随机抽取若干元素，只在这些维度上返回数值梯度。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        # 加上 h
        fxph = f(x)  # evaluate f(x + h)
        # 计算 f(x + h)
        x[ix] = oldval - h  # increment by h
        # 减去 h
        fxmh = f(x)  # evaluate f(x - h)
        # 计算 f(x - h)
        x[ix] = oldval  # reset
        # 复位

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (
            abs(grad_numerical) + abs(grad_analytic)
        )
        print(
            "numerical: %f analytic: %f, relative error: %e"
            % (grad_numerical, grad_analytic, rel_error)
        )
