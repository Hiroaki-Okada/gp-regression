# デバッグ用
import pdb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

np.random.seed(0)


def rf_regression(train_x, train_y, test_x, n_estimators=1000, max_features=1.0, max_depth=None):
    """
    学習データを用いてテストデータにおける出力を予測するランダムフォレスト回帰を行う.
    具体的には, テストデータにおける出力の予測分布を1つずつ計算していく.

    Parameters
    ----------
    train_x, train_y : ndarray
        学習データの入出力
    test_x : ndarray
        テストデータの入力
    n_estimators : int
        ランダムフォレスト回帰における決定木の数
    max_features : ['sqrt', 'log2', None], int of float, default=1.0
        決定木に用いる特徴量の数
    max_depth : int, default=None
        決定木の深さの最大値

    Returns
    ----------
    mean : ndarray
        ランダムフォレスト回帰による予測値ベクトル
    variance : ndarray
        ランダムフォレスト回帰における各決定木の予測値の分散ベクトル
    """

    train_x_rf = train_x.reshape(-1, 1)
    test_x_rf = test_x.reshape(-1, 1)

    model = RandomForestRegressor(n_estimators=n_estimators,
                                  max_features=max_features,
                                  max_depth=max_depth)

    model.fit(train_x_rf, train_y)

    mean = model.predict(test_x_rf)

    tree_prediction_l = []
    tree_num = model.n_estimators
    for tree in range(tree_num):
        tree_estimates = model.estimators_[tree].predict(test_x_rf)
        tree_prediction_l.append(tree_estimates)

    variance = np.var(tree_prediction_l, axis=0)

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
    ランダムフォレスト回帰の結果を可視化し, テストデータにおける真の出力と比較する.

    Parameters
    ----------
    train_x, train_y, test_x, test_y : ndarray
        学習データの入出力, テストデータの入出力
    mean : ndarray
        ランダムフォレスト回帰で得た予測分布の平均ベクトル
    variance : ndarray
        ランダムフォレスト回帰で得た予測分布の分散ベクトル
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
    plt.title('Random forest regression', fontsize=20)
    plt.legend(loc='lower left', fontsize=20)
    plt.tight_layout()
    plt.show()
    # plt.savefig('Random_forest_regression.jpeg')
    plt.close()

    
def run(n=100):
    """
    学習データ (train_x, train_y) を用いてテストデータにおける出力を予測するランダムフォレスト回帰を行う.
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

    mean, variance = rf_regression(train_x, train_y, test_x)
    plot_prediction(train_x, train_y, test_x, test_y, mean, variance)


if __name__ == '__main__':
    run()