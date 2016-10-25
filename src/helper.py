import numpy as np


def ncc(x, y):
    numerator = sum((x - np.mean(x)) * (y - np.mean(y)))
    denominator = np.sqrt(sum(np.power(x - np.mean(x), 2))) * np.sqrt(sum(np.power(y - np.mean(y), 2)))
    return numerator/denominator
