from __future__ import print_function

import os
from builtins import range
from builtins import object
import numpy as np
from ..classifiers.softmax import *
from past.builtins import xrange


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
        self,
        X,
        y,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.
        用随机梯度下降训练这个线性分类器。

        Inputs:
        输入：
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - X: 形状为 (N, D) 的 numpy 数组，存放训练数据；共有 N 个训练样本，每个样本的维度为 D。
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - y: 形状为 (N,) 的 numpy 数组，存放训练标签；y[i] = c 表示 X[i] 的标签为 0 <= c < C，共 C 个类别。
        - learning_rate: (float) learning rate for optimization.
        - learning_rate: (float) 优化时使用的学习率。
        - reg: (float) regularization strength.
        - reg: (float) 正则化强度。
        - num_iters: (integer) number of steps to take when optimizing
        - num_iters: (integer) 优化时执行的步数
        - batch_size: (integer) number of training examples to use at each step.
        - batch_size: (integer) 每一步使用的训练样本数量。
        - verbose: (boolean) If true, print progress during optimization.
        - verbose: (boolean) 若为 true，则在优化过程中打印进度。

        Outputs:
        输出：
        A list containing the value of the loss function at each training iteration.
        一个列表，包含每次训练迭代时损失函数的值。
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        # 假设 y 的取值为 0...K-1，其中 K 是类别数
        if self.W is None:
            # lazily initialize W
            # 延迟初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        # 运行随机梯度下降来优化 W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            # TODO:                                                                 #
            # 从训练数据中采样 batch_size 个样本及其对应的标签，用于这一轮梯度下降。 #
            # 把数据存到 X_batch，把对应的标签存到 y_batch；采样后 X_batch 的形状应为 (batch_size, dim)， #
            # y_batch 的形状应为 (batch_size,)                                      #
            #                                                                       #
            # 提示：用 np.random.choice 生成索引。有放回采样比无放回采样更快。      #
            #########################################################################


            # evaluate loss and gradient
            # 计算损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            # 执行参数更新
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            # TODO:                                                                 #
            # 用梯度和学习率更新权重。                                              #
            #########################################################################


            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        用这个线性分类器训练好的权重为数据点预测标签。

        Inputs:
        输入：
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - X: 形状为 (N, D) 的 numpy 数组，存放训练数据；共有 N 个训练样本，每个样本的维度为 D。

        Returns:
        返回：
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        - y_pred: X 中数据的预测标签。y_pred 是长度为 N 的一维数组，
          每个元素是给出预测类别的整数。
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        # TODO:                                                                   #
        # 实现这个方法。把预测出的标签存到 y_pred 中。                            #
        ###########################################################################

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        计算损失函数及其导数。
        Subclasses will override this.
        子类会重写这个方法。

        Inputs:
        输入：
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - X_batch: 形状为 (N, D) 的 numpy 数组，包含一个小批量的 N 个数据点；每个点的维度为 D。
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - y_batch: 形状为 (N,) 的 numpy 数组，包含这个小批量数据的标签。
        - reg: (float) regularization strength.
        - reg: (float) 正则化强度。

        Returns: A tuple containing:
        返回：一个元组，包含：
        - loss as a single float
        - loss：单个浮点数
        - gradient with respect to self.W; an array of the same shape as W
        - 关于 self.W 的梯度；与 W 形状相同的数组
        """
        pass

    def save(self, fname):
      # 保存模型参数。
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = {"W": self.W}
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      # 加载模型参数。
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.W = params["W"]
        print(fname, "loaded.")
        return True


class LinearSVM(LinearClassifier):
    # 使用多类 SVM 损失函数的子类
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    # 使用 Softmax + 交叉熵损失函数的子类
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
