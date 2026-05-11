import numpy as np

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def linear(z):
    return z
    
def dlinear(z):
    return np.ones_like(z)

ACTIVATIONS = {
    'relu':    (relu, drelu),
    'softmax': (softmax, None),
    'linear':  (linear, dlinear),
}