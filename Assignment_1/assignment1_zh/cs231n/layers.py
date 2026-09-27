from builtins import range
import numpy as np

# import numexpr as ne # ~~DELETE LINE~~
# 把 numexpr 引入为 ne（此行应删除）


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    计算仿射（全连接）层的前向传播。

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    输入 x 的形状为 (N, d_1, ..., d_k)，包含由 N 个样本组成的小批量，
    其中每个样本 x[i] 的形状为 (d_1, ..., d_k)。我们会把每个输入
    重新整形为维度 D = d_1 * ... * d_k 的向量，然后把它变换为
    维度为 M 的输出向量。

    Inputs:
    输入：
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - x: 存放输入数据的 numpy 数组，形状为 (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - w: 权重 numpy 数组，形状为 (D, M)
    - b: A numpy array of biases, of shape (M,)
    - b: 偏置 numpy 数组，形状为 (M,)

    Returns a tuple of:
    返回一个元组：
    - out: output, of shape (N, M)
    - out: 输出，形状为 (N, M)
    - cache: (x, w, b)
    - cache: 缓存 (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    # TODO:                                                                   #
    # 实现仿射层的前向传播。把结果存到 out 中。
    # 你需要把输入重新整形为若干行。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    计算仿射层的反向传播。

    Inputs:
    输入：
    - dout: Upstream derivative, of shape (N, M)
    - dout: 上游导数，形状为 (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)
    - cache: 元组：
      - x: 输入数据，形状为 (N, d_1, ... d_k)
      - w: 权重，形状为 (D, M)
      - b: 偏置，形状为 (M,)

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dx: 关于 x 的梯度，形状为 (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - dw: 关于 w 的梯度，形状为 (D, M)
    - db: Gradient with respect to b, of shape (M,)
    - db: 关于 b 的梯度，形状为 (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    # TODO:                                                                   #
    # 实现仿射层的反向传播。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    计算修正线性单元（ReLU）层的前向传播。

    Input:
    输入：
    - x: Inputs, of any shape
    - x: 输入，可以是任意形状

    Returns a tuple of:
    返回一个元组：
    - out: Output, of the same shape as x
    - out: 输出，形状与 x 相同
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    # TODO:                                                                   #
    # 实现 ReLU 的前向传播。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    计算修正线性单元（ReLU）层的反向传播。

    Input:
    输入：
    - dout: Upstream derivatives, of any shape
    - dout: 上游导数，可以是任意形状
    - cache: Input x, of same shape as dout
    - cache: 输入 x，形状与 dout 相同

    Returns:
    返回：
    - dx: Gradient with respect to x
    - dx: 关于 x 的梯度
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    # TODO:                                                                   #
    # 实现 ReLU 的反向传播。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    批归一化（batch normalization）的前向传播。

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.
    训练时，样本均值和（未修正的）样本方差由 minibatch 统计量计算得到，
    并用来归一化输入数据。训练时我们还会维护每个特征的均值和方差的
    指数衰减滑动平均，这些平均值在测试时用来归一化数据。

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    每个时间步，我们都基于 momentum 参数用指数衰减来更新均值和方差的滑动平均：

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.
    注意，batch normalization 论文给出的测试阶段行为有所不同：它对每个特征
    用大量训练图像来计算样本均值和方差，而不是使用滑动平均。在本实现中，
    我们选择改用滑动平均，因为它不需要额外的估计步骤；torch7 的
    batch normalization 实现同样使用滑动平均。

    Input:
    输入：
    - x: Data of shape (N, D)
    - x: 形状为 (N, D) 的数据
    - gamma: Scale parameter of shape (D,)
    - gamma: 缩放参数，形状为 (D,)
    - beta: Shift paremeter of shape (D,)
    - beta: 平移参数，形状为 (D,)
    - bn_param: Dictionary with the following keys:
    - bn_param: 包含以下键的字典：
      - mode: 'train' or 'test'; required
      - mode: 'train' 或 'test'；必需
      - eps: Constant for numeric stability
      - eps: 用于数值稳定的常量
      - momentum: Constant for running mean / variance.
      - momentum: 用于 running mean / variance 的常量。
      - running_mean: Array of shape (D,) giving running mean of features
      - running_mean: 形状为 (D,) 的数组，给出各特征的滑动均值
      - running_var Array of shape (D,) giving running variance of features
      - running_var: 形状为 (D,) 的数组，给出各特征的滑动方差

    Returns a tuple of:
    返回一个元组：
    - out: of shape (N, D)
    - out: 形状为 (N, D)
    - cache: A tuple of values needed in the backward pass
    - cache: 反向传播所需的值组成的元组
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        # TODO:                                                               #
        # 实现 batch norm 训练阶段的前向传播。
        # 用 minibatch 的统计量计算均值和方差，用这些统计量对输入数据做归一化，
        # 再用 gamma 和 beta 对归一化后的数据做缩放和平移。
        #
        # 应该把输出存到变量 out 中。反向传播需要的任何中间量都要存到
        # cache 变量中。
        #
        # 还应该用计算出的样本均值和方差配合 momentum 变量，来更新滑动均值
        # running_mean 和滑动方差 running_var，把结果存在这两个变量里。
        #
        # 注意：虽然你要维护的是滑动方差（running variance），但归一化数据时
        # 应当使用标准差（方差的平方根）！
        # 参考原始论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。
        #######################################################################
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        # TODO:                                                               #
        # 实现 batch normalization 测试阶段的前向传播。
        # 用滑动均值和滑动方差对输入数据做归一化，
        # 再用 gamma 和 beta 对归一化后的数据做缩放和平移。
        # 把结果存到 out 变量中。
        #######################################################################
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    # 把更新后的滑动均值写回 bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    批归一化的反向传播。

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    在本实现中，你应该在纸上画出 batch normalization 的计算图，
    然后让梯度沿中间节点反向传播。

    Inputs:
    输入：
    - dout: Upstream derivatives, of shape (N, D)
    - dout: 上游导数，形状为 (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    - cache: 来自 batchnorm_forward 的中间量。

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # TODO:                                                                   #
    # 实现 batch normalization 的反向传播。把结果存到 dx、dgamma 和
    # dbeta 变量中。
    # 参考原始论文 (https://arxiv.org/abs/1502.03167) 可能会有帮助。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    批归一化的另一种反向传播实现。

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.
    在本实现中，你应该在纸上推导 batch normalization 反向传播的导数，
    并尽可能化简。你应该能推导出反向传播的一个简洁表达式。
    更多提示见 jupyter notebook。

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.
    注意：本实现应当接收与 batchnorm_backward 相同的 cache 变量，
    但可能用不到 cache 中的所有值。

    Inputs / outputs: Same as batchnorm_backward
    输入 / 输出：与 batchnorm_backward 相同
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    # TODO:                                                                   #
    # 实现 batch normalization 的反向传播。把结果存到 dx、dgamma 和
    # dbeta 变量中。
    #
    # 在算出关于中心化输入（centered inputs）的梯度之后，
    # 你应该能用一条语句算出关于输入的梯度；
    # 我们的实现只占一行 80 个字符。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.
    layer normalization 的前向传播。

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    在训练和测试阶段，输入数据都按数据点（per data-point）归一化，
    然后用与 batch normalization 相同的 gamma 和 beta 参数做缩放和平移。

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.
    注意，与 batch normalization 不同，layer normalization 在训练和测试阶段的
    行为完全一致，也不需要维护任何形式的滑动平均。

    Input:
    输入：
    - x: Data of shape (N, D)
    - x: 形状为 (N, D) 的数据
    - gamma: Scale parameter of shape (D,)
    - gamma: 缩放参数，形状为 (D,)
    - beta: Shift paremeter of shape (D,)
    - beta: 平移参数，形状为 (D,)
    - ln_param: Dictionary with the following keys:
    - ln_param: 包含以下键的字典：
        - eps: Constant for numeric stability
        - eps: 用于数值稳定的常量

    Returns a tuple of:
    返回一个元组：
    - out: of shape (N, D)
    - out: 形状为 (N, D)
    - cache: A tuple of values needed in the backward pass
    - cache: 反向传播所需的值组成的元组
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    # TODO:                                                                   #
    # 实现 layer norm 训练阶段的前向传播。
    # 对输入数据做归一化，再用 gamma 和 beta 对归一化后的数据做缩放和平移。
    # 提示：可以稍微修改你训练阶段实现的 batch normalization，
    # 再插入一两行恰到好处的代码来完成。特别地，你能想到什么矩阵变换，
    # 使你能够把 batch norm 的代码几乎原样搬过来使用吗？
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.
    layer normalization 的反向传播。

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.
    在本实现中，你可以大量复用你为 batch normalization 已经完成的工作。

    Inputs:
    输入：
    - dout: Upstream derivatives, of shape (N, D)
    - dout: 上游导数，形状为 (N, D)
    - cache: Variable of intermediates from layernorm_forward.
    - cache: 来自 layernorm_forward 的中间量。

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dx: 关于输入 x 的梯度，形状为 (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dgamma: 关于缩放参数 gamma 的梯度，形状为 (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    - dbeta: 关于平移参数 beta 的梯度，形状为 (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    # TODO:                                                                   #
    # 实现 layer norm 的反向传播。
    #
    # 提示：可以稍微修改你训练阶段实现的 batch normalization。
    # 前向传播的提示同样适用！
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.
    执行（inverted）dropout 的前向传播。

    Inputs:
    输入：
    - x: Input data, of any shape
    - x: 输入数据，可以是任意形状
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.
    - dropout_param: 包含以下键的字典：
      - p: Dropout 参数。以概率 p 保留每个神经元的输出。
      - mode: 'test' 或 'train'。若 mode 为 train，则执行 dropout；
        若 mode 为 test，则直接返回输入。
      - seed: 随机数生成器的种子。传入 seed 会让该函数具有确定性，
        这在梯度检查中需要，但在真实网络中不需要。

    Outputs:
    输出：
    - out: Array of the same shape as x.
    - out: 与 x 形状相同的数组。
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    - cache: 元组 (dropout_param, mask)。训练模式下 mask 是用于乘输入的
      dropout 掩码；测试模式下 mask 为 None。

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.
    注意：请实现 **inverted** dropout，而不是普通版的 dropout。
    更多细节见 http://cs231n.github.io/neural-networks-2/#reg。

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    注意 2：请记住 p 是**保留**一个神经元输出的概率；
    这可能与某些资料相反，那些资料把它当作丢弃神经元输出的概率。
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # TODO:                                                               #
        # 实现 inverted dropout 训练阶段的前向传播。
        # 把 dropout 掩码存到 mask 变量中。
        pass
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # TODO:                                                               #
        # 实现 inverted dropout 测试阶段的前向传播。
        pass
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.
    执行（inverted）dropout 的反向传播。

    Inputs:
    输入：
    - dout: Upstream derivatives, of any shape
    - dout: 上游导数，可以是任意形状
    - cache: (dropout_param, mask) from dropout_forward.
    - cache: 来自 dropout_forward 的 (dropout_param, mask)。
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # TODO:                                                               #
        # 实现 inverted dropout 训练阶段的反向传播
        pass
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    卷积层前向传播的朴素实现。

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.
    输入由 N 个数据点组成，每个数据点有 C 个通道、高度 H 和宽度 W。
    我们用 F 个不同的滤波器分别与每个输入做卷积，每个滤波器
    覆盖全部 C 个通道，高度为 HH、宽度为 WW。

    Input:
    输入：
    - x: Input data of shape (N, C, H, W)
    - x: 形状为 (N, C, H, W) 的输入数据
    - w: Filter weights of shape (F, C, HH, WW)
    - w: 形状为 (F, C, HH, WW) 的滤波器权重
    - b: Biases, of shape (F,)
    - b: 偏置，形状为 (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    - conv_param: 包含以下键的字典：
      - 'stride': 相邻感受野在水平和垂直方向上的像素间隔。
      - 'pad': 用于对输入做零填充的像素数。


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.
    填充时，'pad' 个零应当沿输入的高度轴和宽度轴对称放置（即两侧数量相等）。
    注意不要直接修改原始输入 x。

    Returns a tuple of:
    返回一个元组：
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    - out: 输出数据，形状为 (N, F, H', W')，其中 H' 和 W' 由下式给出
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    # TODO:                                                                   #
    # 实现卷积层的前向传播。
    # 提示：填充（padding）可以使用 np.pad 函数。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    卷积层反向传播的朴素实现。

    Inputs:
    输入：
    - dout: Upstream derivatives.
    - dout: 上游导数。
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    - cache: 与 conv_forward_naive 中相同的 (x, w, b, conv_param) 元组

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to x
    - dx: 关于 x 的梯度
    - dw: Gradient with respect to w
    - dw: 关于 w 的梯度
    - db: Gradient with respect to b
    - db: 关于 b 的梯度
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    # TODO:                                                                   #
    # 实现卷积层的反向传播。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    最大池化层前向传播的朴素实现。

    Inputs:
    输入：
    - x: Input data, of shape (N, C, H, W)
    - x: 形状为 (N, C, H, W) 的输入数据
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    - pool_param: 包含以下键的字典：
      - 'pool_height': 每个池化区域的高度
      - 'pool_width': 每个池化区域的宽度
      - 'stride': 相邻池化区域之间的距离

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0
    这里不需要填充，例如你可以假设：

    Returns a tuple of:
    返回一个元组：
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - out: 输出数据，形状为 (N, C, H', W')，其中 H' 和 W' 由下式给出
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    # TODO:                                                                   #
    # 实现最大池化层的前向传播
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    最大池化层反向传播的朴素实现。

    Inputs:
    输入：
    - dout: Upstream derivatives
    - dout: 上游导数
    - cache: A tuple of (x, pool_param) as in the forward pass.
    - cache: 与前向传播中相同的 (x, pool_param) 元组。

    Returns:
    返回：
    - dx: Gradient with respect to x
    - dx: 关于 x 的梯度
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    # TODO:                                                                   #
    # 实现最大池化层的反向传播
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.
    计算空间批归一化（spatial batch normalization）的前向传播。

    Inputs:
    输入：
    - x: Input data of shape (N, C, H, W)
    - x: 形状为 (N, C, H, W) 的输入数据
    - gamma: Scale parameter, of shape (C,)
    - gamma: 缩放参数，形状为 (C,)
    - beta: Shift parameter, of shape (C,)
    - beta: 平移参数，形状为 (C,)
    - bn_param: Dictionary with the following keys:
    - bn_param: 包含以下键的字典：
      - mode: 'train' or 'test'; required
      - mode: 'train' 或 'test'；必需
      - eps: Constant for numeric stability
      - eps: 用于数值稳定的常量
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - momentum: 用于 running mean / variance 的常量。momentum=0 表示
        每个时间步都完全丢弃旧信息，而 momentum=1 表示
        永远不纳入新信息。大多数情况下 momentum=0.9 的默认值效果都很好。
      - running_mean: Array of shape (D,) giving running mean of features
      - running_mean: 形状为 (D,) 的数组，给出各特征的滑动均值
      - running_var Array of shape (D,) giving running variance of features
      - running_var: 形状为 (D,) 的数组，给出各特征的滑动方差

    Returns a tuple of:
    返回一个元组：
    - out: Output data, of shape (N, C, H, W)
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: Values needed for the backward pass
    - cache: 反向传播所需的值
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    # TODO:                                                                   #
    # 实现空间批归一化的前向传播。
    #
    # 提示：可以通过调用你上面实现的普通版 batch normalization 来实现
    # 空间批归一化。
    # 你的实现应该非常短；我们的实现不到五行。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.
    计算空间批归一化的反向传播。

    Inputs:
    输入：
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: Values from the forward pass
    - cache: 前向传播得到的值

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dgamma: 关于缩放参数的梯度，形状为 (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    - dbeta: 关于平移参数的梯度，形状为 (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    # TODO:                                                                   #
    # 实现空间批归一化的反向传播。
    #
    # 提示：可以通过调用你上面实现的普通版 batch normalization 来实现
    # 空间批归一化。
    # 你的实现应该非常短；我们的实现不到五行。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.
    计算空间组归一化（spatial group normalization）的前向传播。
    与 layer normalization 不同，group normalization 把数据中的每个条目
    切成 G 个连续的部分，然后分别独立地对它们做归一化。
    之后对数据施加逐特征的平移和缩放，方式与 batch normalization
    和 layer normalization 完全相同。

    Inputs:
    输入：
    - x: Input data of shape (N, C, H, W)
    - x: 形状为 (N, C, H, W) 的输入数据
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - gamma: 缩放参数，形状为 (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - beta: 平移参数，形状为 (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - G: 要分成的组数（整数），应当是 C 的约数
    - gn_param: Dictionary with the following keys:
    - gn_param: 包含以下键的字典：
      - eps: Constant for numeric stability
      - eps: 用于数值稳定的常量

    Returns a tuple of:
    返回一个元组：
    - out: Output data, of shape (N, C, H, W)
    - out: 输出数据，形状为 (N, C, H, W)
    - cache: Values needed for the backward pass
    - cache: 反向传播所需的值
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    # TODO:                                                                   #
    # 实现空间组归一化的前向传播。
    # 它与 layer norm 的实现极其相似。
    # 特别地，想一想你可以怎样变换矩阵，
    # 使得大部分代码既能与训练阶段的 batch normalization 相似，
    # 也能与 layer normalization 相似！
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.
    计算空间组归一化的反向传播。

    Inputs:
    输入：
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - dout: 上游导数，形状为 (N, C, H, W)
    - cache: Values from the forward pass
    - cache: 前向传播得到的值

    Returns a tuple of:
    返回一个元组：
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dx: 关于输入的梯度，形状为 (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dgamma: 关于缩放参数的梯度，形状为 (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    - dbeta: 关于平移参数的梯度，形状为 (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    # TODO:                                                                   #
    # 实现空间组归一化的反向传播。
    # 它与 layer norm 的实现极其相似。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.
    计算多类 SVM 分类的损失和梯度。

    Inputs:
    输入：
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - x: 形状为 (N, C) 的输入数据，其中 x[i, j] 是第 i 个输入
      在第 j 个类别上的得分。
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    - y: 形状为 (N,) 的标签向量，其中 y[i] 是 x[i] 的标签，
      且 0 <= y[i] < C

    Returns a tuple of:
    返回一个元组：
    - loss: Scalar giving the loss
    - loss: 标量，表示损失
    - dx: Gradient of the loss with respect to x
    - dx: 损失关于 x 的梯度
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    # TODO:                                                                   #
    # 把你 A1 中的解答复制过来。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    计算 softmax 分类的损失和梯度。

    Inputs:
    输入：
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - x: 形状为 (N, C) 的输入数据，其中 x[i, j] 是第 i 个输入
      在第 j 个类别上的得分。
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    - y: 形状为 (N,) 的标签向量，其中 y[i] 是 x[i] 的标签，
      且 0 <= y[i] < C

    Returns a tuple of:
    返回一个元组：
    - loss: Scalar giving the loss
    - loss: 标量，表示损失
    - dx: Gradient of the loss with respect to x
    - dx: 损失关于 x 的梯度
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from A1.
    # TODO:                                                                   #
    # 把你 A1 中的解答复制过来。
    ###########################################################################

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
