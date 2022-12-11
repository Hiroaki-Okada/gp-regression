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


def white_kernel(x1, x2, scale=0.1):
    delta = 1 if x1 == x2 else 2
    return scale * delta


def mean_function(x):
    return np.zeros(len(x))


def kernel_function(x, kernel_func=rbf_kernel, white_noise=False):
    n = x.shape[0]
    kernel = np.empty((n, n))

    for i in range(n):
        for j in range(i, n):
            if white_noise:
                kij = kernel_func(x[i], x[j]) + white_kernel(x[i], x[j])
            else:
                kij = kernel_func(x[i], x[j])

            kernel[i, j] = kij
            kernel[j, i] = kij

    return kernel


def gp_sampling(x, sample_size=5):
    mean = mean_function(x)
    gram_matrix = kernel_function(x)

    return np.random.multivariate_normal(mean, gram_matrix, sample_size)


def visualize_1d(x, f):
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
    x = np.linspace(-5, 5, 100)
    f = gp_sampling(x)
    visualize_1d(x, f)


def sample_2d_func():
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 20)
    xx, yy = np.meshgrid(x, y)

    # xy = np.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    xy = np.c_[xx.reshape(-1), yy.reshape(-1)]

    f = gp_sampling(xy, sample_size=1).reshape(xx.shape)
    visualize_2d(xx, yy, f)


def run():
    sample_1d_func()
    sample_2d_func()


if __name__ == '__main__':
    run()
