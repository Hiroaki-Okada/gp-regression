# デバッグ用
import pdb

import numpy as np


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
    return outputscale * (1 + np.sqrt(3) * dist / lengthscale) * np.exp(-np.sqrt(3) * dist / lengthscale)


# Matern5 カーネル
def matern5_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * (1 + np.sqrt(5) * dist / lengthscale + (5 * dist ** 2) / (3 * lengthscale ** 2)) * np.exp(-np.sqrt(5) * dist / lengthscale)


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