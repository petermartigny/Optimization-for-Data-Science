# Optimization
Optimization for Data Science Course @ Télécom ParisTech, [Alexandre Gramfort](http://alexandre.gramfort.net/) & [Stéphane Gaïffas](http://www.cmap.polytechnique.fr/~gaiffas/)

This course covers a general review of theory and practise of gradient-based algorithms to solve empirical risk minimization problems (mainly linear regression, logistic regression and support vector machines). For all methods, it covers also the proximal approach, dealing with regularization.

## First order algorithms
Study & implementation of ISTA and FISTA algotithms: the first is a vanilla gradient descent algorithm, the second is its accelerated version

## Coordinate descent
The gradient may be update coordinate after coordinate, making the convergence faster using smart updates

## Conjugate gradient descent
An iterative method to solve linear problems with positive definite matrices.
Specific result for quadratic case: it converges in at most n iterations

## Quasi-Newton methods
These methods leverage the Taylor expansion around the optimal to approach the hessian for approaching Newton Method (which does not scale, as such)

## Stochastic Gradient Descent
Instead of updating over the whole dataset, update only on one data point, whosen sequentially and randomly, or on a mini-batch. The SGD performs very well for the first iteration, but then have a bad behavior. Indeed, the estimator is unbiased, but has a large variance.

## SGD with variance reduction
Monte Carlo methods are useful to reduce the variance induced by the SGD algorithm. We review (theoretically and in practise) several such algorithms: SAG, SAGA, SVRG

## Introduction to Non Convex Optimization
Review of several non convex regularization
Intro to conditional gradient algorithm
