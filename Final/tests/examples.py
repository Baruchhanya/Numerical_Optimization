import numpy as np

# Quadratic function with Q = [[1, 0], [0, 1]] (contour lines are circles)
def quadratic_circle(x, hessian=False):
    Q = np.array([[1, 0], [0, 1]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hessian:
        h = 2 * Q
        return f, g, h
    return f, g

# Quadratic function with Q = [[1, 0], [0, 100]] (contour lines are axis-aligned ellipses)
def quadratic_ellipse(x, hessian=False):
    Q = np.array([[1, 0], [0, 100]])
    f = x.T @ Q @ x
    g = 2 * Q @ x
    if hessian:
        h = 2 * Q
        return f, g, h
    return f, g

# Quadratic function with rotated ellipse
def quadratic_rotated_ellipse(x, hessian=False):
    Q1 = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Q2 = np.array([[100, 0], [0, 1]])
    Q3 = Q1.T
    Q = np.dot(Q1, np.dot(Q2, Q3))
    f = np.dot(x.T, np.dot(Q, x))
    g = 2 * np.dot(Q, x)
    if hessian:
        h = 2 * Q
        return f, g, h
    return f, g

# Rosenbrock function
def rosenbrock(x, hessian=False):
    x1, x2 = x
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    g = np.array([-400 * x1 * (x2 - x1**2) - 2 * (1 - x1), 200 * (x2 - x1**2)])
    if hessian:
        h = np.array([[1200 * x1**2 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]])
        return f, g, h
    return f, g

# Linear function
def linear_function(x, hessian=False):
    a = np.array([1, 3])
    f = a.T @ x
    g = a
    if hessian:
        h = np.zeros((2, 2))
        return f, g, h
    return f, g

# Exponential function
def exponential_function(x, hessian=False):
    x1, x2 = x
    f = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    g = np.array([np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1),
                  3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1)])
    if hessian:
        h = np.array([[np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1),
                       3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1)],
                      [3*np.exp(x1 + 3*x2 - 0.1) - 3*np.exp(x1 - 3*x2 - 0.1),
                       9*np.exp(x1 + 3*x2 - 0.1) + 9*np.exp(x1 - 3*x2 - 0.1)]])
        return f, g, h
    return f, g
