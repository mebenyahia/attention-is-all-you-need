import matplotlib.pyplot as plt
import numpy as np
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm


def generate_noisy_points(
    n=10, noise_variance=1e-6, func=np.sin, domain=(-3, 3), seed=42
):
    np.random.seed(seed)

    X = np.random.uniform(domain[0], domain[1], (n, 1))

    y = func(X).flatten() + np.random.randn(n) * noise_variance**0.5

    return X, y


def plot_kernels(X1, X2, kernel):
    """Plot the prior and posterior distributions for a given kernel

    Args:
        X1 (numpy.ndarray): Training points
        X2 (numpy.ndarray): Test points
        y1 (numpy.ndarray): Labels (targets in lecture) of X1
        y2 (numpy.ndarray): Labels (targets in lecture) of X2
        kernel (sklearn.gaussian_process.kernels or custom): Kernel function

    """

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))

    kernel_matrix_train = kernel(X1, X1)
    kernel_matrix_test = kernel(X1, X2)

    # Plot the prior distribution
    ax1.imshow(kernel_matrix_train, cmap="viridis")
    ax1.set_title("Kernel of Observations")
    ax1.set_axis_off()

    ax2.imshow(kernel_matrix_test, cmap="viridis")
    ax2.set_title("Kernel of Observations vs. To-Predict")
    ax2.set_axis_off()

    plt.tight_layout()
    plt.show()


def plot_conditioned_new_visualization(ax, ax2, sampled_points, pdf, X, y):
    """Plot the conditioned visualization

    Args:
        NOTE: The following parameters are a recommendation, feel free to change them

        ax (matplotlib.axes.Axes): The axes to plot to
        ax2 (matplotlib.axes.Axes): The axes to plot to
        sampled_points (numpy.ndarray): The sampled points
        pdf (numpy.ndarray): The probability density function
        X  (numpy.ndarray): The X values
        y  (numpy.ndarray): The y values

    """
    for i in range(5):
        ax.scatter([1, 2], sampled_points[:, i], c="b", s=50)
        ax.plot([1, 2], sampled_points[:, i], c="b", linestyle="--")

    distribution = pdf[X == 1]

    ax.set_title("New Visualization")
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("$y$")
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticks([1, 2])
    ax.set_xlim(0, 3)

    ax2.plot(distribution, y)
    ax2.set_title("Conditioned Probability Density Function")
    ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax2.set_ylim(-3, 3)


def plot_new_visualization(ax, sampled_points):
    ax.scatter([1, 2], sampled_points[:, 0], c="b", s=50)

    ax.plot([1, 2], sampled_points[:, 0], c="b", linestyle="--")

    ax.set_title("New Visualization")
    ax.set_xlabel("Variable Index")
    ax.set_ylabel("$y$")
    ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
    ax.set_xticks([1, 2])
    ax.set_xlim(0, 3)


def plot_mvn(
    ax,
    mean,
    cov,
    distr,
    domain=(-3, 3),
    num=200,
    cmap="viridis",
    sampled_points=None,
    conditioned=False,
):
    """Plot a 2D Gaussian distribution"""
    ax.set_title(
        r"$\mu = [{}, {}], \Sigma = \begin{{bmatrix}} {}, {} \\ {}, {} \end{{bmatrix}}$".format(
            mean[0], mean[1], cov[0, 0], cov[0, 1], cov[1, 0], cov[1, 1]
        )
    )

    x = np.linspace(domain[0] * cov[0, 0], domain[1] * cov[0, 0], num=num)
    y = np.linspace(domain[0] * cov[1, 1], domain[1] * cov[1, 1], num=num)
    X, Y = np.meshgrid(x, y)

    # Generating the density function
    # for each point in the meshgrid
    pdf = distr.pdf(X, Y)

    divider = make_axes_locatable(ax)
    ax.set_aspect("equal")
    ax_x = divider.append_axes("bottom", 1.0, pad=0.9, sharex=ax)
    ax_x.set_title(r"$p(x_1) = \int p(x_1,x_2)dx_2$")
    ax_y = divider.append_axes("right", 1.0, pad=0.5, sharey=ax)
    ax_y.set_title(r"$p(x_2) = \int p(x_1,x_2)dx_1$")

    ax.contour(X, Y, pdf, cmap=cmap)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")

    ax_x.axis("equal")
    ax_y.axis("equal")

    if sampled_points is not None:
        ax.scatter(sampled_points[0, :], sampled_points[1, :], c="r", s=50)
        if conditioned:
            for i in range(5):
                ax.scatter(sampled_points[0, i], sampled_points[1, i], c="b", s=75)
        else:
            ax.scatter(sampled_points[0, 0], sampled_points[1, 0], c="b", s=75)

    # marginals
    pdfx = norm.pdf(x, loc=0, scale=np.sqrt(cov[0, 0]))
    ax_x.plot(x, pdfx)
    pdfy = norm.pdf(y, loc=0, scale=np.sqrt(cov[1, 1]))
    ax_y.plot(pdfy, y)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])


def plot_gp_results(f_sin, X1, X2, y1, y2, domain, pred, limits=(-3, 3)):
    """Plot the postior distribution and some samples"""

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 6))
    # Plot the distribution of the function (mean, covariance)
    ax1.plot(X2, f_sin(X2), "b--", label="$sin(x)$")
    ax1.fill_between(
        X2.flat,
        pred[0] - pred[1],
        pred[0] + pred[1],
        color="red",
        alpha=0.15,
        label="$\\sigma$",
    )
    # ax1.fill_between(
    #     X2.flat,
    #     pred_noisy[0] - pred_noisy[1],
    #     pred_noisy[0] + pred_noisy[1],
    #     color="red",
    #     alpha=0.15,
    #     label="$2\sigma_{2|1}$",
    # )
    ax1.plot(X2, pred[0], "r-", lw=2, label="$\\mu$")
    ax1.plot(X1, y1, "ko", linewidth=2, label="$(x_i, t_i)$")
    ax1.set_xlabel("$x$", fontsize=13)
    ax1.set_ylabel("$y$", fontsize=13)
    ax1.set_title("Distribution of posterior and prior data")
    ax1.axis(
        [
            domain[0],
            domain[1],
            np.min(f_sin(X2)) - (0.5 * np.max(f_sin(X2))),
            np.max(f_sin(X2)) + (0.5 * np.max(f_sin(X2))),
        ]
    )
    ax1.legend(loc="lower left")
    ax1.set_ylim(limits)

    # Plot some samples from this function
    ax2.plot(X2, y2.T, marker="o", linestyle="", markersize=3)
    ax2.set_xlabel("$x$", fontsize=13)
    ax2.set_ylabel("$y$", fontsize=13)
    ax2.set_title("5 different distributions from posterior")
    ax2.axis(
        [
            domain[0],
            domain[1],
            np.min(f_sin(X2)) - (0.5 * np.max(f_sin(X2))),
            np.max(f_sin(X2)) + (0.5 * np.max(f_sin(X2))),
        ]
    )
    ax2.fill_between(
        X2.flat,
        pred[0] - pred[1],
        pred[0] + pred[1],
        color="red",
        alpha=0.15,
        label="$\\sigma_{2|1}$",
    )
    # ax2.fill_between(
    #     X2.flat,
    #     pred_noisy[0] - pred_noisy[1],
    #     pred_noisy[0] + pred_noisy[1],
    #     color="red",
    #     alpha=0.15,
    #     label="$2\sigma_{2|1}$",
    # )
    ax2.set_ylim(limits)
    plt.tight_layout()
    # plt.savefig("gp.png", dpi=300, transparent=True)
    plt.show()
