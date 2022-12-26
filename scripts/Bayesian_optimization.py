# デバッグ用
import pdb

import math
import warnings

import numpy as np
from scipy.stats import norm
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

    """
    isNegative = variance < 0
    if np.any(isNegative):
        warnings.warn('The predicted variance is less than 0. '
                      'These values are set to 0.')
        variance = np.where(variance >= 0, variance, 0.0)
    """

    return mean, variance


# EI 関数
def ei(train_y, test_x, mean, variance, jitter=0.01):
    best_y = np.max(train_y)
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - best_y - jitter) / stdev
    imp = mean - best_y - jitter

    ei = imp * norm.cdf(z) + stdev * norm.pdf(z)

    return ei


# PI 関数
def pi(train_y, test_x, mean, variance, jitter=0.01):
    best_y = np.max(train_y)
    stdev = np.sqrt(variance) + 1e-6

    z = (mean - best_y - jitter) / stdev
    cdf = norm.cdf(z)

    return cdf


# UCB 関数
def ucb(train_y, test_x, mean, variance, jitter=0.01, delta=0.05):
    stdev = np.sqrt(variance) + 1e-6

    dim = 1
    iters = len(train_y)

    beta = np.sqrt(2 * np.log(dim * (iters**2) * (math.pi**2) / (6*delta)))
    ucb = mean + beta * stdev

    return ucb


def evaluate_acq_function(train_y, test_x, mean, variance, acq_func=ei):
    """
    test_x の各要素における獲得関数の値を計算する.

    Parameters
    ----------
    train_y : ndarray
        学習データの出力
    test_x : ndarray
        テストデータの入力
    mean : ndarray
        ガウス過程回帰で得た予測分布の平均ベクトル
    variance : ndarray
        ガウス過程回帰で得た予測分布の分散ベクトル
    acq_val : xxx
        計算する獲得関数の種類
        xxx は ei, pi, ucb から選ぶ

    Returns
    ----------
    ndarray
        計算された獲得関数の値
    """

    return acq_func(train_y, test_x, mean, variance)


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


def plot_prediction_and_acq(train_x, train_y, test_x, test_y, mean, variance, acq_val):
    """
    ガウス過程回帰の結果を可視化し, テストデータにおける真の出力と比較する.
    また, 獲得関数の値も可視化し, 獲得関数が最大の入力（=次に観測値を得るべき入力）を提案する.

    Parameters
    ----------
    train_x, train_y, test_x, test_y : ndarray
        学習データの入出力, テストデータの入出力
    mean : ndarray
        ガウス過程回帰で得た予測分布の平均ベクトル
    variance : ndarray
        ガウス過程回帰で得た予測分布の分散ベクトル

    acq_val : ndarray
        test_x の各要素における獲得関数の値のベクトル
    """

    best_idx = np.argmax(acq_val)
    next_x = test_x[best_idx]
    max_acq_val = acq_val[best_idx]

    fig = plt.figure(figsize = [15, 20])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(test_x, test_y, 'x', color='green', label='True data', lw=2)
    ax1.plot(train_x, train_y, 'o', color='crimson', label='Training data', lw=2)

    std = np.sqrt(variance)
    ax1.plot(test_x, mean, color='blue', label='Mean')
    ax1.fill_between(test_x, mean + std*2, mean - std*2, alpha=0.2, color='blue', label='Standard deviation')

    ax1.set_xlabel('x', fontsize=20)
    ax1.set_ylabel('y', fontsize=20)
    ax1.tick_params(labelsize=20)
    ax1.set_title('Bayesian optimization', fontsize=20)
    ax1.legend(loc='lower left', fontsize=20)

    ax2.plot(test_x, acq_val, color='blue')
    ax2.scatter(next_x, max_acq_val, c='crimson', linewidths=10)
    ax2.set_xlabel('x', fontsize=20)
    ax2.set_ylabel('acq val', fontsize=20)
    ax2.tick_params(labelsize=20)
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    plt.tight_layout()
    plt.show()
    # plt.savefig('Bayesian_optimization.jpeg')
    plt.close()


def run(n=100):
    """
    学習データ (train_x, train_y) を用いてテストデータにおける出力を予測するガウス過程回帰を行う.
    また, 回帰の結果を可視化して予測精度を検証する.
    続いて, 獲得関数の値を計算し, 次に観測値を得るべき入力点を提案する.

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

    acq_val = evaluate_acq_function(train_y, test_x, mean, variance)
    plot_prediction_and_acq(train_x, train_y, test_x, test_y, mean, variance, acq_val)


if __name__ == '__main__':
    run()