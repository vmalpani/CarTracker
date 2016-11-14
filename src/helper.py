import numpy as np
import cv2
from scipy import signal
from PIL import Image


def load_image(image_fh, color=True):
    """loads image for caffe net"""
    im = Image.open(image_fh)
    im.load()

    # caffe wants rgb values as floats 0-1 rather than ints 0-255
    img = np.array(im, dtype=np.float32) / 255
    im.close()
    del im
    if img.ndim == 2:
        # convert single-channel grayscale to rgb
        img = img[:, :, np.newaxis]
        if color:
            img = np.tile(img, (1, 1, 3))
    elif img.shape[2] == 4:
        # strip out alpha if it's there
        img = img[:, :, :3]

    if len(img.shape) != 3 or img.shape[2] != 3:
        raise RuntimeError('image has bad shape: %s' % str(img.shape))

    return img


def expand_search_window(img, rect, search_window_size=120):
    """
    img: input image
    rect: detection window in last frame
    search_window_size: search window size to augment the detection window

    Given a detection from previous frame generated the search window for
    current frame
    """
    center_x = rect[0] + rect[2]/2
    center_y = rect[1] + rect[3]/2

    search_window_x = max(0, center_x - search_window_size/2)
    search_window_y = max(0, center_y - search_window_size/2)

    search_window_max_x = min(img.shape[1],
                              search_window_x + search_window_size)
    search_window_max_y = min(img.shape[0],
                              search_window_y + search_window_size)

    width = search_window_max_x - search_window_x
    height = search_window_max_y - search_window_y
    return (search_window_x, search_window_y, width, height)


def ncc(x, y):
    """calculates normalized cross correlation between two histograms"""
    numerator = sum((x - np.mean(x)) * (y - np.mean(y)))
    term1 = np.sqrt(sum(np.power(x - np.mean(x), 2)))
    term2 = np.sqrt(sum(np.power(y - np.mean(y), 2)))
    return abs(numerator/(term1*term2))


def chi_squared(x, y):
    """calculates chi squared similarity between two histograms"""
    num = (x - y) ** 2
    denom = x + y
    denom[denom == 0] = np.infty
    frac = num / denom
    chi_sqr = 0.5 * np.sum(frac)
    similarity = 1 / (chi_sqr + 1.0e-4)
    return similarity


def compute_epanechnikov_kernel(ht, wt):
    """epanechnikov kernel for given template size to improve tracking"""
    kernel = np.zeros((ht, wt))
    for i in range(1, ht+1):
        for j in range(1, wt+1):
            tmp = (float(i - ht/2.0)/ht)**2 + (float(j - wt/2.0)/wt)**2
            kernel[i-1][j-1] = 1 - tmp**2 if tmp < 1 else 0
    return kernel


def draw_bbox(fname, image, bboxs):
    """draws bbox on image and writes it to disk"""
    bboxs = [box for box in bboxs if box]
    for (x, y, w, h) in bboxs:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(fname, image)


def median_filtering(img):
    """perform median filtering to smooth an image"""
    smooth_img = np.zeros(img.shape)
    smooth_img[:, :, 0] = signal.medfilt2d(img[:, :, 0])
    smooth_img[:, :, 1] = signal.medfilt2d(img[:, :, 1])
    smooth_img[:, :, 2] = signal.medfilt2d(img[:, :, 2])
    return smooth_img


def sliding_window(image, search_window, stride, template_size):
    """performs sliding window sweep yielding next window"""
    max_y = min(search_window[1] + search_window[3], image.shape[0]) \
        - template_size[1]
    max_x = min(search_window[0] + search_window[2], image.shape[1]) \
        - template_size[0]
    for y in xrange(search_window[1], max_y, stride):
        for x in xrange(search_window[0], max_x, stride):
            assert y + template_size[1] < image.shape[0]
            assert x + template_size[0] < image.shape[1]
            yield (x, y, image[y:y + template_size[1], x:x + template_size[0]])
