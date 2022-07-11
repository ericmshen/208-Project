import numpy as np
import opendp

def gaussian(shift=0., scale=1., size=None):
    return np.random.normal(loc=shift, scale=scale, size=size)

def clamp(x, bounds):
    return np.clip(x, *bounds)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def laplace(shift = 0.0, scale = 1.0, size = None):
    return np.random.laplace(loc=shift, scale=scale, size=size)

def ols_regression(x, y):
    s_xy = np.sum(x * y)
    s_xx = np.sum(x ** 2)
    return s_xy / s_xx

def dp_ols_regression(x, y, epsilon, b):
    s_xy_hat = np.sum(x * y) + laplace(scale = 4*b*b / epsilon)
    s_xx_hat = s_xx_hat = np.sum(x ** 2) + laplace(scale = 2*b*b / epsilon)
    while s_xx_hat == 0.:
        s_xx_hat = np.sum(x ** 2) + laplace(scale = 2*b*b / epsilon)
    return s_xy_hat / s_xx_hat

def bounded_mean(x, bounds):
    x_clamped = clamp(x, bounds)
    return np.mean(x_clamped)

def release_dp_mean(x, bounds, epsilon, delta=1e-6, mechanism="laplace"):
    """Release a DP mean. 
    Assumes that the dataset size n is public information.
    """
    sensitive_mean = bounded_mean(x, bounds)

    n = len(x)
    lower, upper = bounds
    # Sensitivity in terms of an absolute distance metric
    # Both the laplace and gaussian mechanisms can noise queries
    #    with sensitivities expressed in absolute distances
    sensitivity = (upper - lower) / n
    
    if mechanism == "laplace":
        scale = sensitivity / epsilon
        dp_mean = sensitive_mean + laplace(scale=scale)
    elif mechanism == "gaussian":
        scale = (sensitivity / epsilon) * np.sqrt(2*np.log(2/delta)) 
        dp_mean = sensitive_mean + gaussian(scale=scale)
    else:
        raise ValueError(f"unrecognized mechanism: {mechanism}")

    return dp_mean

def bootstrap(x, n):
    """Sample n values with replacement from n."""
    index = np.random.randint(low=0., high=len(x), size=n)
    return x[index]

import pandas as pd
data = pd.read_csv(
    "https://raw.githubusercontent.com/privacytoolsproject/cs208/master/data/MaPUMS5full.csv")

# define public information
n = len(data)            # in this case, dataset length is considered public, and is not protected
educ_bounds = (1., 16.)  # easily guessable without looking at the data
data = data[['educ', 'married']].values.astype(float)