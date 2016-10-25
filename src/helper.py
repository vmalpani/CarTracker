import numpy as np
import cv2


def ncc(x, y):
    numerator = sum((x - np.mean(x)) * (y - np.mean(y)))
    denominator = np.sqrt(sum(np.power(x - np.mean(x), 2))) * np.sqrt(sum(np.power(y - np.mean(y), 2)))
    return numerator/denominator

def compute_epanechnikov_kernel(ht, wt):
    kernel = np.zeros((ht, wt))
    for i in range(1, ht+1):
        for j in range(1, wt+1):
            tmp = (float(i - ht/2.0)/ht)**2 + (float(j - wt/2.0)/wt)**2
            kernel[i-1][j-1] = 1 - tmp**2 if tmp < 1 else 0
    return kernel

def draw_bbox(fname, image, (x, y, w, h)):
    cv2.rectangle(image, (x, y), (x+w, y+h),(0, 255, 0), 2)
    cv2.imwrite(fname, image)
