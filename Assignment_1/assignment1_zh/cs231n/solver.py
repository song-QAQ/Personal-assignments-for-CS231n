from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training classification
    models. The Solver performs stochastic gradient descent using different
    update rules defined in optim.py.
    Solver 封装了训练分类模型所需的全部逻辑。Solver 使用 optim.py 中定义的不同更新规则
    来执行随机梯度下降。

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data to watch out for overfitting.
    solver 同时接收训练数据和验证数据及其标签，因此它可以定期检查模型在训练集和验证集上的
    分类准确率，以便及时发现过拟合。

    To train a model, you will first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. You will then call the train() method to run the optimization
    procedure and train the model.
    要训练一个模型，首先构造 Solver 实例，把模型、数据集以及各种选项（学习率、批大小等）
    传给构造函数；然后调用 train() 方法运行优化过程来训练模型。

    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.
    train() 方法返回后，model.params 中包含训练过程中在验证集上表现最好的参数。
    此外，实例变量 solver.loss_history 是一个列表，保存训练过程中遇到的所有损失值；
    实例变量 solver.train_acc_history 和 solver.val_acc_history 是列表，
    分别记录每个 epoch 时模型在训练集和验证集上的准确率。

    Example usage might look something like this:
    用法示例大致如下：

    data = {
      'X_train': # training data
      'y_train': # training labels
      'X_val': # validation data
      'y_val': # validation labels
    }
    model = MyAwesomeModel(hidden_size=100, reg=10)
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=5, batch_size=200,
                    print_every=100)
    solver.train()


    A Solver works on a model object that must conform to the following API:
    Solver 作用的模型对象必须符合以下 API：

    - model.params must be a dictionary mapping string parameter names to numpy
      arrays containing parameter values.
    - model.params 必须是一个字典，把字符串形式的参数名映射到存放参数值的 numpy 数组。

    - model.loss(X, y) must be a function that computes training-time loss and
      gradients, and test-time classification scores, with the following inputs
      and outputs:
    - model.loss(X, y) 必须是一个函数，用来计算训练时的损失和梯度、以及测试时的分类得分，
      其输入和输出如下：

      Inputs:
      输入：
      - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
      - X: 一个 minibatch 的输入数据，形状为 (N, d_1, ..., d_k)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].
      - y: 形状为 (N,) 的标签数组，给出 X 的标签，其中 y[i] 是 X[i] 的标签。

      Returns:
      返回：
      If y is None, run a test-time forward pass and return:
      如果 y 为 None，则执行测试时的前向传播并返回：
      - scores: Array of shape (N, C) giving classification scores for X where
        scores[i, c] gives the score of class c for X[i].
      - scores: 形状为 (N, C) 的数组，给出 X 的分类得分，
        其中 scores[i, c] 是 X[i] 属于类别 c 的得分。

      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      如果 y 不为 None，则执行训练时的前向和反向传播，并返回一个元组：
      - loss: Scalar giving the loss
      - loss: 标量，给出损失值
      - grads: Dictionary with the same keys as self.params mapping parameter
        names to gradients of the loss with respect to those parameters.
      - grads: 键与 self.params 相同的字典，把参数名映射到损失关于这些参数的梯度。
    """

    def __init__(self, model, data, **kwargs):
        """
        Construct a new Solver instance.
        构造一个新的 Solver 实例。

        Required arguments:
        必需参数：
        - model: A model object conforming to the API described above
        - model: 符合上述 API 的模型对象
        - data: A dictionary of training and validation data containing:
          'X_train': Array, shape (N_train, d_1, ..., d_k) of training images
          'X_val': Array, shape (N_val, d_1, ..., d_k) of validation images
          'y_train': Array, shape (N_train,) of labels for training images
          'y_val': Array, shape (N_val,) of labels for validation images
        - data: 包含训练和验证数据的字典：
          'X_train': 训练图像数组，形状为 (N_train, d_1, ..., d_k)
          'X_val': 验证图像数组，形状为 (N_val, d_1, ..., d_k)
          'y_train': 训练图像的标签数组，形状为 (N_train,)
          'y_val': 验证图像的标签数组，形状为 (N_val,)

        Optional arguments:
        可选参数：
        - update_rule: A string giving the name of an update rule in optim.py.
          Default is 'sgd'.
        - update_rule: 字符串，给出 optim.py 中某个更新规则的名称。默认是 'sgd'。
        - optim_config: A dictionary containing hyperparameters that will be
          passed to the chosen update rule. Each update rule requires different
          hyperparameters (see optim.py) but all update rules require a
          'learning_rate' parameter so that should always be present.
        - optim_config: 一个字典，包含要传给所选更新规则的那些超参数。每个更新规则需要不同的
          超参数（见 optim.py），但所有更新规则都需要 'learning_rate' 参数，
          因此它必须始终存在。
        - lr_decay: A scalar for learning rate decay; after each epoch the
          learning rate is multiplied by this value.
        - lr_decay: 学习率衰减的标量；每经过一个 epoch，学习率都乘以这个值。
        - batch_size: Size of minibatches used to compute loss and gradient
          during training.
        - batch_size: 训练时用于计算损失和梯度的 minibatch 的大小。
        - num_epochs: The number of epochs to run for during training.
        - num_epochs: 训练中要运行的 epoch 数量。
        - print_every: Integer; training losses will be printed every
          print_every iterations.
        - print_every: 整数；每 print_every 次迭代打印一次训练损失。
        - verbose: Boolean; if set to false then no output will be printed
          during training.
        - verbose: 布尔值；若设为 false，训练期间不打印任何输出。
        - num_train_samples: Number of training samples used to check training
          accuracy; default is 1000; set to None to use entire training set.
        - num_train_samples: 用于检查训练准确率的训练样本数；默认是 1000；
          设为 None 表示使用整个训练集。
        - num_val_samples: Number of validation samples to use to check val
          accuracy; default is None, which uses the entire validation set.
        - num_val_samples: 用于检查验证准确率的验证样本数；默认是 None，即使用整个验证集。
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch.
        - checkpoint_name: 如果不是 None，则每个 epoch 把模型 checkpoint 保存到这个名字。
        """
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        # Unpack keyword arguments
        # 解包关键字参数
        self.update_rule = kwargs.pop("update_rule", "sgd")
        self.optim_config = kwargs.pop("optim_config", {})
        self.lr_decay = kwargs.pop("lr_decay", 1.0)
        self.batch_size = kwargs.pop("batch_size", 100)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_train_samples = kwargs.pop("num_train_samples", 1000)
        self.num_val_samples = kwargs.pop("num_val_samples", None)

        self.checkpoint_name = kwargs.pop("checkpoint_name", None)
        self.print_every = kwargs.pop("print_every", 10)
        self.verbose = kwargs.pop("verbose", True)

        # Throw an error if there are extra keyword arguments
        # 如果有多余的关键字参数就报错
        if len(kwargs) > 0:
            extra = ", ".join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Make sure the update rule exists, then replace the string
        # name with the actual function
        # 确认该更新规则存在，然后用实际的函数替换字符串名称
        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        为优化过程设置一些记录用的变量。不要手动调用。
        """
        # Set up some variables for book-keeping
        # 设置一些用于记录的变量
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        # Make a deep copy of the optim_config for each parameter
        # 为每个参数深拷贝一份 optim_config
        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    def _step(self):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        执行一次梯度更新。它由 train() 调用，不应手动调用。
        """
        # Make a minibatch of training data
        # 取一个 minibatch 的训练数据
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        # Compute loss and gradient
        # 计算损失和梯度
        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # Perform a parameter update
        # 执行一次参数更新
        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    def _save_checkpoint(self):
        if self.checkpoint_name is None:
            return
        checkpoint = {
            "model": self.model,
            "update_rule": self.update_rule,
            "lr_decay": self.lr_decay,
            "optim_config": self.optim_config,
            "batch_size": self.batch_size,
            "num_train_samples": self.num_train_samples,
            "num_val_samples": self.num_val_samples,
            "epoch": self.epoch,
            "loss_history": self.loss_history,
            "train_acc_history": self.train_acc_history,
            "val_acc_history": self.val_acc_history,
        }
        filename = "%s_epoch_%d.pkl" % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """
        Check accuracy of the model on the provided data.
        检查模型在给定数据上的准确率。

        Inputs:
        输入：
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - X: 数据数组，形状为 (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - y: 标签数组，形状为 (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - num_samples: 如果不是 None，就对数据做子采样，只在 num_samples 个数据点上测试模型。
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.
        - batch_size: 把 X 和 y 按这个大小分批，避免占用过多内存。

        Returns:
        返回：
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        - acc: 标量，给出被模型正确分类的样本所占的比例。
        """

        # Maybe subsample the data
        # 可能对数据做子采样
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        # 分批计算预测结果
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    def train(self):
        """
        Run optimization to train the model.
        运行优化来训练模型。
        """
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            # Maybe print training loss
            # 可能打印训练损失
            if self.verbose and t % self.print_every == 0:
                print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, num_iterations, self.loss_history[-1])
                )

            # At the end of every epoch, increment the epoch counter and decay
            # the learning rate.
            # 每个 epoch 结束时，epoch 计数器加一，并衰减学习率。
            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]["learning_rate"] *= self.lr_decay

            # Check train and val accuracy on the first iteration, the last
            # iteration, and at the end of each epoch.
            # 在第一次迭代、最后一次迭代以及每个 epoch 结束时检查训练和验证准确率。
            first_it = t == 0
            last_it = t == num_iterations - 1
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(
                    self.X_train, self.y_train, num_samples=self.num_train_samples
                )
                val_acc = self.check_accuracy(
                    self.X_val, self.y_val, num_samples=self.num_val_samples
                )
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print(
                        "(Epoch %d / %d) train acc: %f; val_acc: %f"
                        % (self.epoch, self.num_epochs, train_acc, val_acc)
                    )

                # Keep track of the best model
                # 记录表现最好的模型
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        # At the end of training swap the best params into the model
        # 训练结束时，把最好的参数换进模型
        self.model.params = self.best_params
