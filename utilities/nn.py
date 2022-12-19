import numpy as np

def one_hot_encode(y) -> np.ndarray:
    """One hot encode utility tool"""
    encoded = np.zeros((len(y), 10))

    for idx, val in enumerate(y):
        encoded[idx][val] = 1

    return encoded
