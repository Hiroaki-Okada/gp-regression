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
    return np.exp(-dist / lengthscale)


# Matern3 カーネル
def matern3_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return (1 + np.sqrt(3) * dist / lengthscale) * \
        np.exp(-np.sqrt(3) * dist / lengthscale)


# Matern5 カーネル
def matern5_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return (1 + np.sqrt(5) * dist / lengthscale + (5 * dist ** 2) / (3 * lengthscale ** 2)) * \
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


def gp_regression(train_x, train_y, test_x, kernel_func=rbf_kernel, white_noise=True):
    """
    学習データを用いてテストデータにおける出力を予測するガウス過程回帰を行う.
    具体的には, テストデータにおける出力の予測分布を1つずつ計算していく.
    予測分布の計算は, 学習データとテストデータの同時分布から学習データを周辺化した
    条件付き分布を求めることで実行できる.

    Parameters
    ----------
    train_x, train_y : ndarray
        学習データの入出力
    test_x : ndarray
        テストデータの入力
    kernel_func : xxx_kernel
        使用するカーネルの関数名
        xxx は linear, polynomial, matern1, matern3, matern5, rbf から選ぶ
    white_noise : bool
        観測誤差としてホワイトノイズを考慮する

    Returns
    ----------
    mean : ndarray
        ガウス過程回帰で得た予測分布の平均ベクトル
    variance : ndarray
        ガウス過程回帰で得た予測分布の分散ベクトル
    """

    n = train_x.shape[0]
    K = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            x1 = train_x[i]
            x2 = train_x[j]

            if white_noise:
                kij = kernel_func(x1, x2) + white_kernel(i, j)
            else:
                kij = kernel_func(x1, x2)

            K[i, j] = kij
            K[j, i] = kij

    m = test_x.shape[0]
    mean = np.empty(m)
    variance = np.empty(m)

    for i in range(m):
        k1_T = np.empty(n)
        for j in range(n):
            k1_T[j] = kernel_func(train_x[j], test_x[i])

        if white_kernel:
            k2 = kernel_func(test_x[i], test_x[i]) + white_kernel(i, j)
        else:
            k2 = kernel_func(test_x[i], test_x[i])

        mean_i = k1_T @ np.linalg.inv(K) @ train_y
        variance_i = k2 - k1_T @ np.linalg.inv(K) @ k1_T.T

        mean[i] = mean_i
        variance[i] = variance_i

    return mean, variance


def plot_train_and_test(train_x, train_y, test_x, test_y):
    """
    学習データとテストデータを可視化する.
    """

    plt.figure(figsize=(15, 8))
    plt.plot(test_x, test_y, 'x', color='green', label='True data', lw=2)
    plt.plot(train_x, train_y, 'o', color='crimson', label='Training data', lw=2)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title('True and training data', fontsize=20)
    plt.legend(loc='lower left', fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig('True_and_training_data.jpeg')
    plt.close()


def plot_prediction(train_x, train_y, test_x, test_y, mean, variance):
    """
    ガウス過程回帰の結果を可視化し, テストデータにおける真の出力と比較する.

    Parameters
    ----------
    train_x, train_y, test_x, test_y : ndarray
        学習データの入出力, テストデータの入出力
    mean : ndarray
        ガウス過程回帰で得た予測分布の平均ベクトル
    variance : ndarray
        ガウス過程回帰で得た予測分布の分散ベクトル
    """

    plt.figure(figsize=(15, 8))
    plt.plot(test_x, test_y, 'x', color='green', label='True data', lw=2)
    plt.plot(train_x, train_y, 'o', color='crimson', label='Training data', lw=2)

    std = np.sqrt(variance)
    plt.plot(test_x, mean, color='blue', label='Mean')
    plt.fill_between(test_x, mean + std*2, mean - std*2, alpha=0.2, color='blue', label='Standard deviation')

    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title('Gaussian process regression', fontsize=20)
    plt.legend(loc='lower left', fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Gaussian_process_regression.jpeg')
    plt.close()

def run(n=100):
    """
    学習データ (train_x, train_y) を用いてテストデータにおける出力を予測するガウス過程回帰を行う.
    また, 回帰の結果を可視化して予測精度を検証する.

    Parameters
    ----------
    n : int
        テストデータの個数
    """

    test_x = np.linspace(0, 4*np.pi, n)
    test_y = 2*np.sin(test_x) + 3*np.cos(2*test_x) + 5*np.sin(2/3*test_x)

    train_idx = np.random.choice(n, 20, replace=False)
    train_x = test_x[train_idx]
    train_y = test_y[train_idx] + np.random.normal(0, 1, len(train_idx))
    plot_train_and_test(train_x, train_y, test_x, test_y)

    mean, variance = gp_regression(train_x, train_y, test_x)
    plot_prediction(train_x, train_y, test_x, test_y, mean, variance)


if __name__ == '__main__':
    run()