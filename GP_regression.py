import pdb

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


def linear_kernel(x1, x2, variance=1.0):
    return variance * np.dot(x1, x2)


def polynomial_kernel(x1, x2, offset=0, power=2):
    return (np.dot(x1, x2) + offset) ** power


def matern1_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return np.exp(-dist / lengthscale)


def matern3_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return (1 + np.sqrt(3) * dist / lengthscale) * \
        np.exp(-np.sqrt(3) * dist / lengthscale)


def matern5_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return (1 + np.sqrt(5) * dist / lengthscale + (5 * dist ** 2) / (3 * lengthscale ** 2)) * \
        np.exp(-np.sqrt(5) * dist / lengthscale)


def rbf_kernel(x1, x2, lengthscale=1.0, outputscale=1.0):
    dist = np.linalg.norm(x1 - x2)
    return outputscale * np.exp(-0.5 * dist ** 2 / (lengthscale ** 2))


def white_kernel(x1, x2, scale=0.01):
    delta = 1 if x1 == x2 else 0
    return scale * delta


def gp_regression(train_x, train_y, test_x, kernel_func=rbf_kernel, white_noise=True):
    n = train_x.shape[0]
    K = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            x1 = train_x[i]
            x2 = train_x[j]

            if white_noise:
                kij = kernel_func(x1, x2) + white_kernel(x1, x2)
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

        k2 = kernel_func(test_x[i], test_x[i])

        mean_i = k1_T @ np.linalg.inv(K) @ train_y
        variance_i = k2 - k1_T @ np.linalg.inv(K) @ k1_T.T

        mean[i] = mean_i
        variance[i] = variance_i

    return mean, variance


def plot_train_and_test(train_x, train_y, test_x, test_y):
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
