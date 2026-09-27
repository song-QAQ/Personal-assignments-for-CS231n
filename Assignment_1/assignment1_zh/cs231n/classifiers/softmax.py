from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)
    Softmax 损失函数，朴素实现（使用循环）

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.
    输入的维度为 D，共有 C 个类别，运算在包含 N 个样本的小批量上进行。

    Inputs:
    输入：
    - W: A numpy array of shape (D, C) containing weights.
    - W: 形状为 (D, C) 的 numpy 数组，存放权重。
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - X: 形状为 (N, D) 的 numpy 数组，存放一个小批量的数据。
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - y: 形状为 (N,) 的 numpy 数组，存放训练标签；y[i] = c 表示
      X[i] 的标签为 c，其中 0 <= c < C。
    - reg: (float) regularization strength
    - reg: (float) 正则化强度

    Returns a tuple of:
    返回一个元组：
    - loss as single float
    - loss：单个浮点数
    - gradient with respect to weights W; an array of same shape as W
    - 关于权重 W 的梯度；与 W 形状相同的数组
    """
    # Initialize the loss and gradient to zero.
    # 把 loss 和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    # 计算损失和梯度
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        # 用数值稳定的方式计算概率
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        # 归一化
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        # 负对数概率就是损失


    # normalized hinge loss plus regularization
    # 归一化后的 hinge 损失再加上正则化项
    # 注意：上面那句英文注释是官方源码的笔误。本函数计算的是 softmax 交叉熵损失
    # （见上面对 loss 的累加），与 hinge（合页）损失无关。
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    # TODO:                                                                     #
    # 计算损失函数的梯度并存到 dW 中。                                          #
    # 与其先算出损失再求导，不如在计算损失的同时就把导数一起算出来，这样可能更简单。 #
    # 因此你可能需要修改上面的一些代码来计算梯度。                              #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Softmax 损失函数，向量化版本。

    Inputs and outputs are the same as softmax_loss_naive.
    输入和输出与 softmax_loss_naive 相同。
    """
    # Initialize the loss and gradient to zero.
    # 把 loss 和梯度初始化为零。
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    # TODO:                                                                     #
    # 实现 softmax 损失的向量化版本，把结果存到 loss 中。                       #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    # TODO:                                                                     #
    # 实现 softmax 损失梯度的向量化版本，把结果存到 dW 中。                     #
    #                                                                           #
    # 提示：与其从零开始计算梯度，不如复用你在计算损失时用到的中间值，这样可能更容易。 #
    #############################################################################


    return loss, dW
