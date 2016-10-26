import numpy as np
import cv2
from scipy import signal


def ncc(x, y):
    numerator = sum((x - np.mean(x)) * (y - np.mean(y)))
    term1 = np.sqrt(sum(np.power(x - np.mean(x), 2)))
    term2 = np.sqrt(sum(np.power(y - np.mean(y), 2)))
    return abs(numerator/(term1*term2))


def chi_squared(x, y):
    num = (x - y) ** 2
    denom = x + y
    denom[denom == 0] = np.infty
    frac = num / denom
    chi_sqr = 0.5 * np.sum(frac)
    similarity = 1 / (chi_sqr + 1.0e-4)
    return similarity


def compute_epanechnikov_kernel(ht, wt):
    kernel = np.zeros((ht, wt))
    for i in range(1, ht+1):
        for j in range(1, wt+1):
            tmp = (float(i - ht/2.0)/ht)**2 + (float(j - wt/2.0)/wt)**2
            kernel[i-1][j-1] = 1 - tmp**2 if tmp < 1 else 0
    return kernel


def draw_bbox(fname, image, (x, y, w, h)):
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(fname, image)


def median_filtering(img):
    smooth_img = np.zeros(img.shape)
    smooth_img[:, :, 0] = signal.medfilt2d(img[:, :, 0])
    smooth_img[:, :, 1] = signal.medfilt2d(img[:, :, 1])
    smooth_img[:, :, 2] = signal.medfilt2d(img[:, :, 2])
    return smooth_img


def sliding_window(image, stride, window_size):
    for y in xrange(0, image.shape[0]-window_size[1], stride):
        for x in xrange(0, image.shape[1]-window_size[0], stride):
            assert y + window_size[1] < image.shape[0]
            assert x + window_size[0] < image.shape[1]
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def sliding_window_new(image, search_window, stride, template_size):
    max_y = min(search_window[1] + search_window[3], image.shape[0]) - template_size[1]  # noqa
    max_x = min(search_window[0] + search_window[2], image.shape[1]) - template_size[0]  # noqa
    for y in xrange(search_window[1], max_y, stride):
        for x in xrange(search_window[0], max_x, stride):
            assert y + template_size[1] < image.shape[0]
            assert x + template_size[0] < image.shape[1]
            yield (x, y, image[y:y + template_size[1], x:x + template_size[0]])
