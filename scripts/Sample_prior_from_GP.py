# デバッグ用
import pdb

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# 線形カーネル
def linear_kernel(x1, x2, variance=1.0):
    return variance * np.dot(x1, x2)


# 多項式カーネル
def polynomial_kernel(x1, x2, offset=0, power=2):
    return (np.dot(x1, x2) + offset) ** power


# Matern1 カーネル
def matern1_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * np.exp(-dist / lengthscale)


# Matern3 カーネル
def matern3_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * (1 + np.sqrt(3) * dist / lengthscale) * \
                      np.exp(-np.sqrt(3) * dist / lengthscale)


# Matern5 カーネル
def matern5_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * (1 + np.sqrt(5) * dist / lengthscale + (5 * dist ** 2) / (3 * lengthscale ** 2)) * \
                      np.exp(-np.sqrt(5) * dist / lengthscale)


# RBF カーネル
def rbf_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * np.exp(-0.5 * dist ** 2 / (lengthscale ** 2))


def white_kernel(idx1, idx2, scale=0.01):
    """
    White カーネル.
    観測誤差を考慮する場合, カーネルの対角成分に足す小さな値を計算する.

    Parameters
    ----------
    idx1 : int
        カーネル行列の行インデックス
    idx2 : int
        カーネル行列の列インデックス
    scale : float
        スケーリング因子

    Returns
    ----------
    float
        対角成分に足す値
    """

    delta = 1 if idx1 == idx2 else 0
    
    return scale * delta


def mean_function(x):
    """
    平均ベクトルを用意.

    Parameters
    ----------
    x : ndarray
        入力点のデータ
    
    Returns
    ----------
    ndarray
        入力点のデータ数と等しい数の要素を持つゼロベクトル
        ただし, 入力データの平均はあらかじめ 0 にされている必要がある
    """

    return np.zeros(len(x))


def kernel_function(x, kernel_func=rbf_kernel, white_noise=False):
    """
    カーネル行列を用意.

    Parameters
    ----------
    x : ndarray
        入力点のデータ
    kernel_func : xxx_kernel
        使用するカーネルの関数名
        xxx は linear, polynomial, matern1, matern3, matern5, rbf から選ぶ
    white_noise : bool
        観測誤差としてホワイトノイズを考慮する

    Returns
    ----------
    kernel : ndarray
        計算されたカーネル行列
    """
    
    n = x.shape[0]
    kernel = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            if white_noise:
                kij = kernel_func(x[i], x[j]) + white_kernel(i, j)
            else:
                kij = kernel_func(x[i], x[j])

            kernel[i, j] = kij
            kernel[j, i] = kij

    return kernel


def gp_sampling(x, sample_size=5):
    """
    平均ベクトルとカーネル行列を用意し,
    ガウス過程から sample_size 個の関数をサンプリングする.

    Parameters
    ----------
    x : ndarray
        入力点のデータ
    sample_size : int
        ガウス過程からサンプリングする関数の個数

    Returns
    ----------
    ndarray
        サンプリングされた関数の値の ndarray
    """

    mean = mean_function(x)
    gram_matrix = kernel_function(x)

    return np.random.multivariate_normal(mean, gram_matrix, sample_size)


def visualize_1d(x, f):
    """
    1次元空間でガウス過程からサンプリングされた関数を可視化.
    """

    plt.figure(figsize=(10, 8))

    for y in f:
        plt.plot(x, y, lw=2)

    plt.xlabel('x', fontsize=20)
    plt.ylabel('f', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title('Sample prior from GP (1d)', fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Sample_prior_from_GP_1d.jpeg')
    plt.close()


def visualize_2d(x, y, f):
    """
    2次元空間でガウス過程からサンプリングされた関数を可視化.
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, f, cmap='CMRmap', linewidth=0)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('y', fontsize=20)
    ax.set_zlabel('f', fontsize=20)
    ax.set_title('Sample prior from GP (2d)', fontsize=20)
    ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
    # fig.savefig('Sample_prior_from_GP_2d.jpeg')
    plt.close()


def sample_1d_func():
    """
    1次元空間でガウス過程から関数をサンプリングする.
        1. 入力点生成
        2. 平均ベクトルとカーネル行列を用意
        3. 多次元正規分布から関数をサンプリング
        4. 可視化
    """

    x = np.linspace(-5, 5, 100)
    f = gp_sampling(x)
    visualize_1d(x, f)


def sample_2d_func():
    """
    2次元空間でガウス過程から関数をサンプリングする.
    """

    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    xx, yy = np.meshgrid(x, y)

    # xx と yy を1次元ベクトル化した後に結合し, 2次元座標を生成 
    # xy = np.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    xy = np.c_[xx.reshape(-1), yy.reshape(-1)]

    f = gp_sampling(xy, sample_size=1).reshape(xx.shape)
    visualize_2d(xx, yy, f)


def run():
    sample_1d_func()
    sample_2d_func()


if __name__ == '__main__':
    run()