from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.
    给定图像的像素数据和若干可作用于单张图像的特征函数，
    对所有图像分别施加每一个特征函数，把每张图像的特征向量拼接起来，
    并把所有图像的特征存放在一个矩阵中。

    Inputs:
    输入：
    - imgs: N x H X W X C array of pixel data for N images.
    - imgs: N 张图像的像素数据，形状为 N x H x W x C。
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - feature_fns: k 个特征函数组成的列表。第 i 个特征函数应以
      一个 H x W x D 数组作为输入，返回长度为 F_i 的一维数组。
    - verbose: Boolean; if true, print progress.
    - verbose: 布尔值；为真时打印进度。

    Returns:
    返回：
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    形状为 (N, F_1 + ... + F_k) 的数组，其中每一列是
    单张图像所有特征拼接的结果。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    # 用第一张图像确定各特征的维度
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    # 既然已经知道各特征的维度，就可以分配一个大数组，
    # 把所有特征按列存放进去。
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    # 为其余图像提取特征。
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale
    把 RGB 图像转换为灰度图

      Parameters:
      参数：
        rgb : RGB image
        rgb : RGB 图像

      Returns:
      返回：
        gray : grayscale image
        gray : 灰度图

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image
    计算图像的梯度方向直方图（HOG）特征

         Modified from skimage.feature.hog
         修改自 skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
      参考文献：
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
      参数：
        im : an input grayscale or rgb image
        im : 输入的灰度图或 RGB 图像

      Returns:
      返回：
        feat: Histogram of Gradient (HOG) feature
        feat: 梯度方向直方图（HOG）特征

    """

    # convert rgb to grayscale if needed
    # 如有需要，把 RGB 转成灰度图
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # image size
    # 图像尺寸
    orientations = 9  # number of gradient bins
    # 梯度直方图的 bin 数
    cx, cy = (8, 8)  # pixels per cell
    # 每个 cell 的像素数

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    # 计算 x 方向上的梯度
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    # 计算 y 方向上的梯度
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    # 梯度幅值
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation
    # 梯度方向

    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    # x 方向上的 cell 数
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    # y 方向上的 cell 数
    # compute orientations integral images
    # 计算各方向的积分图
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        # 为这个方向创建新的积分图
        # 只保留落在该区间内的方向
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # select magnitudes for those orientations
        # 选出这些方向对应的梯度幅值
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T

    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.
    使用色相（hue）计算图像的颜色直方图。

    Inputs:
    输入：
    - im: H x W x C array of pixel data for an RGB image.
    - im: 一张 RGB 图像的像素数据，形状为 H x W x C。
    - nbin: Number of histogram bins. (default: 10)
    - nbin: 直方图的 bin 数量。（默认值：10）
    - xmin: Minimum pixel value (default: 0)
    - xmin: 像素值下限（默认值：0）
    - xmax: Maximum pixel value (default: 255)
    - xmax: 像素值上限（默认值：255）
    - normalized: Whether to normalize the histogram (default: True)
    - normalized: 是否对直方图做归一化（默认值：True）

    Returns:
    返回：
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
      长度为 nbin 的一维向量，给出输入图像在色相上的颜色直方图。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # return histogram
    # 返回直方图
    return imhist


# ~~START DELETE~~
# These are some other features that we implemented to play around, but aren't
# distributing to students.
# 这些是另外一些我们实现来练手的特征，并没有分发给学生。
def color_histogram(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """Compute color histogram feature for an image
    计算图像的颜色直方图特征

      Parameters:
      参数：
        im : a numpy array of grayscale or rgb image
        im : 灰度图或 RGB 图像的 numpy 数组
        nbin : number of histogram bins (default: 10)
        nbin : 直方图的 bin 数量（默认值：10）
        xmin : minimum pixel value (default: 0)
        xmin : 像素值下限（默认值：0）
        xmax : maximum pixel value (deafult: 255)
        xmax : 像素值上限（默认值：255）
        normalized : bool flag to normalize the histogram
        normalized : 是否对直方图做归一化的布尔标志

      Returns:
      返回：
        feat : color histogram feature
        feat : 颜色直方图特征

    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    # grayscale image
    # 灰度图
    if ndim == 2:
        imhist, bin_edges = np.histogram(im, bins=bins, density=normalized)
        return imhist
    # rgb image
    # RGB 图像
    elif ndim == 3:
        color_hist = np.array([])
        # loop through three color channels
        # 遍历三个颜色通道
        for k in range(3):
            # compute normalized histogram
            # 计算归一化直方图
            imhist, bin_edges = np.histogram(im[:, :, k], bins=bins, density=normalized)
            imhist = imhist * np.diff(bin_edges)
            # concatenate histogram
            # 拼接直方图
            color_hist = np.concatenate((color_hist, imhist))
        # return histogram
        # 返回直方图
        return color_hist
    # unknown image type
    # 未知图像类型
    return np.array([])


def color_histogram_spatial(img, levels=3, nbin=4):
    """
    Color histogram over a pyramid.
    在图像金字塔上计算颜色直方图。
    """
    feats = []

    for level in range(1, levels + 1):
        chunks = np.array_split(img, level, axis=0)
        chunks = [np.array_split(chunk, level, axis=1) for chunk in chunks]
        for x in chunks:
            for chunk in x:
                feats.append(color_histogram_cross(chunk, nbin=nbin))

    return np.hstack(feats)


def color_histogram_cross(img, nbin=5, normalized=True):
    """
    RGB color histogram where our bins are 3 dimensional.
    RGB 颜色直方图，其 bin 是三维的。
    """
    height, width, channels = img.shape
    new_size = (height * width, channels)
    colors = np.reshape(img, new_size)
    return np.histogramdd(colors, bins=nbin, normed=normalized)[0].flatten()


# ~~END DELETE~~
