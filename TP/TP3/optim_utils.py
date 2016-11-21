"""
Example cost functions or objective functions to optimize.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen as rosenbrock
from scipy.optimize import rosen_der as rosenbrock_prime
from scipy.optimize import rosen_hess as rosenbrock_hessian
###############################################################################
# Gaussian functions with varying conditionning


def gaussian(x):
    return np.exp(-np.sum(x**2))


def gaussian_prime(x):
    return -2 * x * np.exp(-np.sum(x**2))


def gaussian_prime_prime(x):
    return -2 * np.exp(-x**2) + 4 * x**2 * np.exp(-x**2)


def mk_gauss(epsilon, ndim=2):
    def f(x):
        x = np.asarray(x)
        y = x.copy()
        y *= np.power(epsilon, np.arange(ndim))
        return -gaussian(.5 * y) + 1

    def f_prime(x):
        x = np.asarray(x)
        y = x.copy()
        scaling = np.power(epsilon, np.arange(ndim))
        y *= scaling
        return -.5 * scaling * gaussian_prime(.5 * y)

    def hessian(x):
        epsilon = .07
        x = np.asarray(x)
        y = x.copy()
        scaling = np.power(epsilon, np.arange(ndim))
        y *= .5 * scaling
        H = -.25 * np.ones((ndim, ndim)) * gaussian(y)
        d = 4 * y * y[:, np.newaxis]
        d.flat[::ndim + 1] += -2
        H *= d
        return H

    return f, f_prime, hessian

###############################################################################
# Quadratic functions with varying conditionning


def mk_quad(epsilon, ndim=2):
    def f(x):
        x = np.asarray(x)
        y = x.copy()
        y *= np.power(epsilon, np.arange(ndim))
        return .33 * np.sum(y**2)

    def f_prime(x):
        x = np.asarray(x)
        y = x.copy()
        scaling = np.power(epsilon, np.arange(ndim))
        y *= scaling
        return .33 * 2 * scaling * y

    def hessian(x):
        scaling = np.power(epsilon, np.arange(ndim))
        return .33 * 2 * np.diag(scaling ** 2)

    return f, f_prime, hessian


###############################################################################
# Super ill-conditionned problem: the Rosenbrock function
# Let's use the Rosenbrock function implemented in scipy

###############################################################################
# Helpers to wrap the functions

class LoggingFunction(object):

    def __init__(self, function, counter=None):
        self.function = function
        if counter is None:
            counter = list()
        self.counter = counter
        self.all_x = list()
        self.all_f_x = list()
        self.counts = list()

    def __call__(self, x0):
        self.all_x.append(x0)
        f_x = self.function(np.asarray(x0))
        self.all_f_x.append(f_x)
        self.counter.append('f')
        self.counts.append(len(self.counter))
        return f_x


class CountingFunction(object):

    def __init__(self, function, counter=None):
        self.function = function
        if counter is None:
            counter = list()
        self.counter = counter

    def __call__(self, x0):
        self.counter.append('f_prime')
        return self.function(x0)


###############################################################################
# A formatter to print values on contours
def super_fmt(value):
    if value > 1:
        if np.abs(int(value) - value) < .1:
            out = '$10^{%.1i}$' % value
        else:
            out = '$10^{%.1f}$' % value
    else:
        value = np.exp(value - .01)
        if value > .1:
            out = '%1.1f' % value
        elif value > .01:
            out = '%.2f' % value
        else:
            out = '%.2e' % value
    return out


def plot_convergence(f, ax, all_x_k, all_f_k, all_x, x_min, x_max,
                     y_min, y_max):
    x, y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    x = x.T
    y = y.T

    levels = dict()

    X = np.concatenate((x[np.newaxis, ...], y[np.newaxis, ...]), axis=0)
    z = np.apply_along_axis(f, 0, X)
    log_z = np.log(z + .01)
    ax.imshow(log_z,
              extent=[x_min, x_max, y_min, y_max],
              cmap=plt.cm.gray_r, origin='lower',
              vmax=log_z.min() + 1.5 * log_z.ptp())
    contours = ax.contour(log_z,
                          levels=levels.get(f, None),
                          extent=[x_min, x_max, y_min, y_max],
                          cmap=plt.cm.gnuplot, origin='lower')
    levels[f] = contours.levels
    ax.clabel(contours, inline=1, fmt=super_fmt, fontsize=14)

    ax.plot(all_x_k[:, 0], all_x_k[:, 1], 'b-', linewidth=2)
    ax.plot(all_x_k[:, 0], all_x_k[:, 1], 'k+')

    ax.plot(all_x[:, 0], all_x[:, 1], 'k.', markersize=4)

    ax.plot([0], [0], 'rx', markersize=12)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.draw()

    plt.tight_layout()
    plt.draw()


def test_solver(optimizer):
    x_min, x_max = -1, 2
    y_min, y_max = 2.25 / 3 * x_min - .2, 2.25 / 3 * x_max - .2
    x_min *= 1.2
    x_max *= 1.2
    y_min *= 1.2
    y_max *= 1.2

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    for index, [ax, (f, f_prime, f_hessian)] in enumerate(zip(axes, (
            mk_quad(.7),
            mk_quad(.02),
            (rosenbrock, rosenbrock_prime, rosenbrock_hessian)))):

        print("\nRunning solver on case %d" % (index + 1))

        # Run optimization method logging all the function calls
        logging_f = LoggingFunction(f)
        x0 = np.array([1.6, 1.1])
        all_x_k, all_f_k = optimizer(x0, logging_f, f_prime, f_hessian)

        # Plot the convergence
        all_x = np.array(logging_f.all_x)
        plot_convergence(
            f, ax, all_x_k, all_f_k, all_x, x_min, x_max, y_min, y_max)

    plt.show()


def test_solver_scipy(optimizer):
    x_min, x_max = -1, 2
    y_min, y_max = 2.25 / 3 * x_min - .2, 2.25 / 3 * x_max - .2
    x_min *= 1.2
    x_max *= 1.2
    y_min *= 1.2
    y_max *= 1.2

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

    for index, [ax, (f, f_prime, f_hessian)] in enumerate(zip(axes, (
            mk_quad(.7),
            mk_quad(.02),
            (rosenbrock, rosenbrock_prime, rosenbrock_hessian)))):

        print("\nRunning solver on case %d" % (index + 1))

        # Run optimization method logging all the function calls
        logging_f = LoggingFunction(f)
        x0 = np.array([1.6, 1.1])
        optimizer(f = logging_f, x0 = x0, fprime = f_prime, jac = f_hessian)#[-1][[0, 4]]

        # Plot the convergence
        all_x = np.array(logging_f.all_x)
        plot_convergence(
            f, ax, all_x_k, all_f_k, all_x, x_min, x_max, y_min, y_max)

    plt.show()