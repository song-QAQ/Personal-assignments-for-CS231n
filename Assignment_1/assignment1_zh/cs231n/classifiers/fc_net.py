from builtins import range
from builtins import object
import os
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    两层全连接神经网络，使用 ReLU 非线性和 softmax 损失，采用模块化的层设计。
    假设输入维度为 D，隐藏层维度为 H，在 C 个类别上进行分类。

    The architecure should be affine - relu - affine - softmax.
    网络结构应为 affine - relu - affine - softmax。

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.
    注意：这个类不实现梯度下降；它改为与一个独立的 Solver 对象交互，
    由该对象负责运行优化。

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    模型的可学习参数存放在字典 self.params 中，它把参数名映射到 numpy 数组。
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.
        初始化一个新的网络。

        Inputs:
        输入：
        - input_dim: An integer giving the size of the input
        - input_dim: 整数，给出输入的大小
        - hidden_dim: An integer giving the size of the hidden layer
        - hidden_dim: 整数，给出隐藏层的大小
        - num_classes: An integer giving the number of classes to classify
        - num_classes: 整数，需要分类的类别数
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - weight_scale: 标量，给出权重随机初始化时使用的标准差。
        - reg: Scalar giving L2 regularization strength.
        - reg: 标量，给出 L2 正则化强度。
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # TODO:                                                                    #
        # 初始化两层网络的权重和偏置。权重应从均值为 0.0、标准差为                 #
        # weight_scale 的高斯分布中初始化，偏置应初始化为零。所有                  #
        # 权重和偏置都应存放在字典 self.params 中：第一层的权重                    #
        # 和偏置使用键 'W1' 和 'b1'，第二层的权重和偏置使用键                      #
        # 'W2' 和 'b2'。                                                           #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.
        计算一个数据小批量的损失和梯度。

        Inputs:
        输入：
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        Returns:
        返回：
        If y is None, then run a test-time forward pass of the model and return:
        若 y 为 None，则运行模型在测试时的前向传播并返回：
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.
        - scores: 形状为 (N, C) 的数组，给出分类得分，其中
          scores[i, c] 是 X[i] 在类别 c 上的分类得分。

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        若 y 不为 None，则运行训练时的前向和反向传播，返回一个元组：
        - loss: Scalar value giving the loss
        - loss: 标量，损失值
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        - grads: 字典，键与 self.params 相同，把参数名映射到
          损失关于这些参数的梯度。
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        # TODO:                                                                    #
        # 实现两层网络的前向传播，计算 X 的分类得分，                              #
        # 并存入 scores 变量。                                                     #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        # 如果 y 为 None，则处于测试模式，直接返回 scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # TODO:                                                                    #
        # 实现两层网络的反向传播。把损失存入 loss 变量，把梯度存入                 #
        # grads 字典。用 softmax 计算数据损失，并确保 grads[k] 保存的是            #
        # self.params[k] 的梯度。不要忘记加上 L2 正则化！                          #
        #                                                                          #
        # 注意：为确保你的实现与我们的实现一致、能通过自动化测试，                 #
        # 请确保你的 L2 正则化包含一个系数 0.5，                                   #
        # 以简化梯度的表达式。                                                     #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      # 保存模型参数。
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.
    多层全连接神经网络。

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be
    该网络包含任意数量的隐藏层、ReLU 非线性和 softmax 损失函数。它还以可选项的形式
    实现 dropout 和批归一化/层归一化。对于有 L 层的网络，结构为

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.
    其中批归一化/层归一化与 dropout 是可选的，{...} 块重复 L - 1 次。

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    可学习参数存放在 self.params 字典中，并通过 Solver 类来学习。
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.
        初始化一个新的 FullyConnectedNet。

        Inputs:
        输入：
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - hidden_dims: 整数列表，给出每个隐藏层的大小。
        - input_dim: An integer giving the size of the input.
        - input_dim: 整数，给出输入的大小。
        - num_classes: An integer giving the number of classes to classify.
        - num_classes: 整数，需要分类的类别数。
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - dropout_keep_ratio: 0 到 1 之间的标量，给出 dropout 强度。
            若 dropout_keep_ratio=1，则网络完全不使用 dropout。
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - normalization: 网络使用哪种归一化。可选值为 "batchnorm"、
            "layernorm"，或 None 表示不做归一化（默认）。
        - reg: Scalar giving L2 regularization strength.
        - reg: 标量，给出 L2 正则化强度。
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - weight_scale: 标量，给出权重随机初始化时使用的标准差。
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - dtype: numpy 数据类型对象；所有计算都用该类型进行。float32 更快但精度较低，
            所以做数值梯度检查时应使用 float64。
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        - seed: 若不为 None，则把这个随机种子传给 dropout 层。
            这会让 dropout 层变得确定，从而可以对模型做梯度检查。
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        # TODO:                                                                    #
        # 初始化网络的参数，把所有值存放在 self.params 字典中。第一层的            #
        # 权重和偏置存到 W1 和 b1；第二层用 W2 和 b2，依此类推。权重应从           #
        # 均值为 0、标准差等于 weight_scale 的正态分布中初始化。偏置应初始化为零。 #
        #                                                                          #
        # 使用批归一化时，第一层的 scale 和 shift 参数存到 gamma1 和 beta1；       #
        # 第二层用 gamma2 和 beta2，依此类推。scale 参数应初始化为 1，             #
        # shift 参数应初始化为 0。                                                 #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        # 使用 dropout 时，需要给每个 dropout 层传一个 dropout_param 字典，
        # 让该层知道 dropout 概率和模式（训练 / 测试）。
        # 可以给每个 dropout 层传同一个 dropout_param。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        # 使用批归一化时，需要记录滑动均值和方差，
        # 因此要给每个批归一化层传一个专门的 bn_param 对象。
        # 第一个批归一化层的前向传播传 self.bn_params[0]，
        # 第二个批归一化层的前向传播传 self.bn_params[1]，依此类推。
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        # 把所有参数转换为正确的数据类型。
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        计算全连接网络的损失和梯度。
        
        Inputs:
        输入：
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - X: 输入数据数组，形状为 (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].
        - y: 标签数组，形状为 (N,)。y[i] 给出 X[i] 的标签。

        Returns:
        返回：
        If y is None, then run a test-time forward pass of the model and return:
        若 y 为 None，则运行模型在测试时的前向传播并返回：
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.
        - scores: 形状为 (N, C) 的数组，给出分类得分，其中
            scores[i, c] 是 X[i] 在类别 c 上的分类得分。

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        若 y 不为 None，则运行训练时的前向和反向传播，返回一个元组：
        - loss: Scalar value giving the loss
        - loss: 标量，损失值
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        - grads: 字典，键与 self.params 相同，把参数名映射到
            损失关于这些参数的梯度。
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        # 为批归一化参数和 dropout 参数设置训练/测试模式，
        # 因为它们在训练和测试时的行为不同。
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        # TODO:                                                                    #
        # 实现全连接网络的前向传播，计算 X 的分类得分，                            #
        # 并存入 scores 变量。                                                     #
        #                                                                          #
        # 使用 dropout 时，需要把 self.dropout_param 传给每个                      #
        # dropout 前向传播。                                                       #
        #                                                                          #
        # 使用批归一化时，需要把 self.bn_params[0] 传给第一个批归一化层的          #
        # 前向传播，把 self.bn_params[1] 传给第二个批归一化层的                    #
        # 前向传播，依此类推。                                                     #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        # 如果是测试模式，提前返回。
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        # TODO:                                                                    #
        # 实现全连接网络的反向传播。把损失存入 loss 变量，把梯度存入               #
        # grads 字典。用 softmax 计算数据损失，并确保 grads[k] 保存的是            #
        # self.params[k] 的梯度。不要忘记加上 L2 正则化！                          #
        #                                                                          #
        # 使用批归一化/层归一化时，不需要对 scale 和 shift 参数做正则化。          #
        #                                                                          #
        # 注意：为确保你的实现与我们的实现一致、能通过自动化测试，                 #
        # 请确保你的 L2 正则化包含一个系数 0.5，                                   #
        # 以简化梯度的表达式。                                                     #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      # 保存模型参数。
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
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
        self.params = params
        print(fname, "loaded.")
        return True