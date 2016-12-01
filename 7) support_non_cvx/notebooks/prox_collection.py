# -*- coding: utf-8 -*-
"""
Created on Thu Oct  28 22:49:12 2016

@author: salmon
"""

import numpy as np
from math import cos, acos, pi

# to avoid issues with zeros values.
eps_precision = 1e-7

###############################################################################
# prox functions


def l0_prox(x, threshold):
    """  hard-thresholding function """
    z = x
    if np.abs(x) < np.sqrt(2 * threshold):
            z = 0
    return z


def l1_prox(x, threshold):
    """  soft-thresholding function """
    y = np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    return y


def l22_prox(x, threshold):
    """  Wiener function """
    y = 1. / (1. + threshold) * x
    return y


def enet_prox(x, threshold, beta=0.5):
    """  Wiener function """
    y = 1. / (1. + beta) * l1_prox(x, threshold)
    return y


def mcp_prox(x, threshold, gamma=1.2):
    """  mcp-proximal operator function, as a constraint gamma >1 """
    y = np.sign(x) * np.maximum(np.abs(x) - threshold, 0) / (1 - 1 / gamma)
    if np.abs(x) > gamma * threshold:
        y = x
    return y


def scad_prox(x, threshold, gamma=2.1):
    """  scad-proximal operator function, as a constraint gamma > 2 """
    y = l1_prox(x, threshold)
    if (np.abs(x) > 2 * threshold):
        if np.abs(x) < gamma * threshold:
            y = ((gamma - 1.) * x -
                 np.sign(x) * gamma * threshold) / (gamma - 2.)
        else:
            y = x
    return y


def log_prox(x, threshold, epsilon=0.5):
    """ log proximal operator function,
    """
    y = np.zeros([5, 1])  # Note that y[4] = 0
    DeltaP = (x - epsilon) ** 2 - 4 * (threshold - x * epsilon)
    if DeltaP >= 0:
        y[0] = ((x - epsilon) + np.sqrt(DeltaP)) / 2
        y[1] = ((x - epsilon) - np.sqrt(DeltaP)) / 2

    DeltaN = (x + epsilon) ** 2 - 4 * (threshold + x * epsilon)
    if DeltaN >= 0:
        y[2] = ((x + epsilon) + np.sqrt(DeltaN)) / 2
        y[3] = ((x + epsilon) - np.sqrt(DeltaN)) / 2

    val = np.zeros([5, 1])
    for i in range(5):
        val[i] = log_objective(y[i], x, threshold, epsilon)

    ind = np.argmin(val)
    # P = y
    p = y[ind]
    return p


def sqrt_prox(x, threshold):
    """  l05 proximal operator """
    z = np.zeros([4, 1])  # Note that z[4] = 0
    if np.abs(x) < eps_precision:
        y = 0
    else:
        if x > 0:
            p = - x  # Cardano's formula as in wikipedia
            q = threshold / 2.
        else:
            p = x  # Cardano's formula as in wikipedia
            q = - threshold / 2.
        p3 = p ** 3
        q12 = q / 2.
        delta = -(4 * p3 + 27 * q ** 2)
        if delta > 0:
            z[0] = cos(acos(-q12 * (27 / (- p3)) ** 0.5) / 3.) \
                * 2 * (-p / 3.) ** 0.5
            z[1] = cos(2 * pi / 3 + acos(-q12 * (27 / (- p3)) ** 0.5) / 3.) * \
                2 * (-p / 3.) ** 0.5
            z[2] = cos(4 * pi / 3 + acos(-q12 * (27 / (- p3)) ** 0.5) / 3.) * \
                2 * (-p / 3.) ** 0.5
        elif delta < 0:
            u = ((-q + (-delta / 27)) / 2.)**(1 / 3)
            v = ((-q + (-delta / 27)) / 2.)**(1 / 3)
            z[0] = u + v
        else:
            z[0] = 3 * q / p
            z[1] = -3 * q / (2 * p)
        val = np.zeros([4, 1])
        for i in range(4):
            val[i] = sqrt_objective(z[i], x, threshold)
        ind = np.argmin(val)
        # P = y
        if x > 0:
            y = (z[ind]) ** 2
        else:
            y = -(z[ind]) ** 2
    # Test:
    # print(z0 ** 3 + p * z0 + q)
    # print(z1 ** 3 + p * z1 + q)
    # print(z2 ** 3 + p * z2 + q)
    return y


###############################################################################
# penalty functions


def l22_pen(x, threshold):
    """ penalty value for l_2^2 regularization"""
    return threshold * x ** 2 / 2


def l1_pen(x, threshold):
    """ penalty value for l1 regularization"""
    return threshold * np.abs(x)


def l0_pen(x, threshold):
    """ penalty value for l0 regularization"""

    if isinstance(x, np.ndarray):
        z = np.ones(x.shape)
        j = np.abs(x) < eps_precision
        z[j] = 0
    else:
        z = 1
        if x == 0:
            z = 0
    return threshold * z


def enet_pen(x, threshold, beta=0.5):
    """ penalty value for enet regularization"""
    return threshold * np.abs(x) + beta * x ** 2 / 2.


def mcp_pen(x, threshold, gamma=1.2):
    """ penalty value for mcp regularization
        Remind that gamma > 1
    """
    if isinstance(x, np.ndarray):
        z = (0.5 * threshold ** 2 * gamma) * np.ones(x.shape)
        j = np.abs(x) < gamma * threshold
        z[j] = threshold * np.abs(x[j]) - x[j] ** 2 / (2 * gamma)
    else:
        z = (0.5 * threshold ** 2 * gamma)
        if np.abs(x) < gamma * threshold:
            z = threshold * np.abs(x) - x ** 2 / (2 * gamma)
    return z


def scad_pen(x, threshold, gamma=2.1):
    """ penalty value for scad regularization
        Remind that gamma > 2
    """
    z = threshold * np.abs(x)
    if isinstance(x, np.ndarray):
        k = (np.abs(x) > threshold) & (np.abs(x) <= gamma * threshold)
        z[k] = (threshold * gamma * np.abs(x[k]) -
                0.5 * (x[k] ** 2 + threshold ** 2)) / (gamma - 1)
        i = (np.abs(x) > gamma * threshold)
        z[i] = threshold ** 2 * (gamma + 1.) / 2.
    else:
        if (np.abs(x) > threshold):
            if (np.abs(x) <= gamma * threshold):
                z = (threshold * gamma * np.abs(x) -
                     0.5 * (x ** 2 + threshold ** 2)) / (gamma - 1)
            else:
                z = threshold ** 2 * (gamma + 1) / 2
    return z


def log_pen(x, threshold, epsilon=0.5):
    """ penalty value for log regularization"""
    return threshold * np.log(1 + np.abs(x) / epsilon)


def sqrt_pen(x, threshold):
    """ penalty value for sqrt regularization"""
    return threshold * np.sqrt(np.abs(x))


###############################################################################
# prox objective functions


def l22_objective(x, y, threshold):
    """ objective function for l_2^2 regularization"""
    return l22_pen(x, threshold) + (x - y) ** 2 / 2


def l1_objective(x, y, threshold):
    """ objective function for l1 regularization"""
    return l1_pen(x, threshold) + (x - y) ** 2 / 2


def l0_objective(x, y, threshold):
    """ objective function for l0 regularization"""
    return l0_pen(x, threshold) + (x - y) ** 2 / 2


def enet_objective(x, y, threshold, beta=0.5):
    """ objective function for l1 regularization"""
    return enet_pen(x, threshold, beta) + (x - y) ** 2 / 2


def mcp_objective(x, y, threshold, gamma=1.2):
    """ objective function for mcp regularization
        Remind that gamma > 1
    """
    return mcp_pen(x, threshold, gamma) + (x - y) ** 2 / 2


def scad_objective(x, y, threshold, gamma=2.1):
    """ objective function for mcp regularization
        Remind that gamma > 2
    """
    return scad_pen(x, threshold, gamma) + (x - y) ** 2 / 2


def log_objective(x, y, threshold, epsilon=0.5):
    """ objective function for log regularization"""
    return log_pen(x, threshold, epsilon) + (x - y) ** 2 / 2


def sqrt_objective(x, y, threshold):
    """ objective function for log regularization"""
    return sqrt_pen(x, threshold) + (x - y) ** 2 / 2
