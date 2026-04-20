import numpy as np

def mu_law(x, mu=255):
    return np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

def mu_law_expansion(x, mu=255.0):
    x = x / mu * 2 - 1
    s = np.sign(x) * (np.expm1(np.abs(x) * np.log1p(mu)) / mu)
    return s

def quantize_data(x, classes=256):
    x = np.clip(x, -1.0, 1.0)
    x_mu = mu_law(x, mu=classes-1)
    q = ((x_mu + 1) / 2 * (classes - 1)).astype(np.int64)
    return q
