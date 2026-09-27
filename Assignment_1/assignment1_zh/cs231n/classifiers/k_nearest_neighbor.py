from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
    # 使用 L2 距离的 kNN 分类器
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.
        训练分类器。对 k 近邻而言，这一步只是把训练数据记下来。

        Inputs:
        输入：
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - X: 形状为 (num_train, D) 的 numpy 数组，存放训练数据，
          包含 num_train 个样本，每个样本的维度为 D。
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        - y: 形状为 (N,) 的 numpy 数组，存放训练标签，
             y[i] 是 X[i] 对应的标签。
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.
        用该分类器预测测试数据的标签。

        Inputs:
        输入：
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - X: 形状为 (num_test, D) 的 numpy 数组，存放测试数据，
             包含 num_test 个样本，每个样本的维度为 D。
        - k: The number of nearest neighbors that vote for the predicted labels.
        - k: 参与投票、决定预测标签的最近邻个数。
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.
        - num_loops: 决定用哪种实现来计算训练点与测试点之间的距离。

        Returns:
        返回：
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        - y: 形状为 (num_test,) 的 numpy 数组，存放测试数据的预测标签，
          y[i] 是测试点 X[i] 的预测标签。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.
        用嵌套循环（同时遍历训练数据和测试数据）计算 X 中每个测试点与
        self.X_train 中每个训练点之间的距离。

        Inputs:
        输入：
        - X: A numpy array of shape (num_test, D) containing test data.
        - X: 形状为 (num_test, D) 的 numpy 数组，存放测试数据。

        Returns:
        返回：
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        - dists: 形状为 (num_test, num_train) 的 numpy 数组，其中 dists[i, j]
          是第 i 个测试点与第 j 个训练点之间的欧氏距离。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                # TODO:                                                             #
                # 计算第 i 个测试点与第 j 个训练点之间的 l2 距离，                    #
                # 把结果存到 dists[i, j]。注意：不要对维度做循环，                     #
                # 也不要用 np.linalg.norm()。                                        #
                #####################################################################
                pass
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.
        用单层循环（只遍历测试数据）计算 X 中每个测试点与 self.X_train 中
        每个训练点之间的距离。

        Input / Output: Same as compute_distances_two_loops
        输入 / 输出：与 compute_distances_two_loops 相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            # TODO:                                                               #
            # 计算第 i 个测试点与所有训练点之间的 l2 距离，                         #
            # 把结果存到 dists[i, :]。                                            #
            # 不要使用 np.linalg.norm()。                                         #
            #######################################################################
            pass
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.
        不使用任何显式循环，计算 X 中每个测试点与 self.X_train 中
        每个训练点之间的距离。

        Input / Output: Same as compute_distances_two_loops
        输入 / 输出：与 compute_distances_two_loops 相同
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        # TODO:                                                                 #
        # 不使用任何显式循环，计算所有测试点与所有训练点之间的 l2 距离，         #
        # 把结果存到 dists 中。                                                 #
        #                                                                       #
        # 这个函数只能用基础的数组运算来实现；                                   #
        # 特别地，不要使用 scipy 里的函数，                                      #
        # 也不要使用 np.linalg.norm()。                                         #
        #                                                                       #
        # 提示：试着用矩阵乘法加两个广播求和来构造 l2 距离。                     #
        #########################################################################

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.
        给定测试点与训练点之间的距离矩阵，为每个测试点预测一个标签。

        Inputs:
        输入：
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.
        - dists: 形状为 (num_test, num_train) 的 numpy 数组，其中 dists[i, j]
          给出第 i 个测试点与第 j 个训练点之间的距离。

        Returns:
        返回：
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        - y: 形状为 (num_test,) 的 numpy 数组，存放测试数据的预测标签，
          y[i] 是测试点 X[i] 的预测标签。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            # 一个长度为 k 的列表，用来存第 i 个测试点的 k 个最近邻的标签。
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            # TODO:                                                                 #
            # 用距离矩阵找出第 i 个测试点的 k 个最近邻，再用 self.y_train            #
            # 查出这些邻居的标签。把这些标签存到 closest_y 里。                     #
            # 提示：查一下 numpy.argsort 这个函数。                                 #
            #########################################################################


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            # TODO:                                                                 #
            # 现在已经找到 k 个最近邻的标签了，你需要找出 closest_y 这个标签        #
            # 列表中出现次数最多的标签，把它存到 y_pred[i]。                        #
            # 出现次数相同时，选择较小的那个标签。                                  #
            #########################################################################


        return y_pred
