import glob
import os

import numpy as np
import cv2

import helper


RESULT_DIR = 'result_histogram'


def generate_histogram(img, (x, y, w, h), bins=32, use_kernel=False):
    """generate image histogram for the specified region"""
    template = img[y:y+h, x:x+w]
    if use_kernel:
        normalized = template / float(bins)
        kernel = helper.compute_epanechnikov_kernel(h, w)
        normalized[:, :, 0] *= kernel
        normalized[:, :, 1] *= kernel
        normalized[:, :, 2] *= kernel
        template = normalized * float(bins)

    car_hist = cv2.calcHist([template], [0], None, [bins], [0, bins])
    car_hist = car_hist.astype(float) / np.sum(car_hist)

    return car_hist


def generate_hog_features(img, (x, y, w, h)):
    """generate hog features for the specified region"""
    hog = cv2.HOGDescriptor()
    car_template = img[y:y+h, x:x+w]
    resized_image = cv2.resize(car_template, (64, 128))
    car_features = hog.compute(np.uint8(resized_image))
    return car_features


def main():
    """main driver function"""

    images = sorted(glob.glob('data/*.jpg'))
    first_img = cv2.imread(images[0])
    bbox = [6, 166, 43, 27]

    helper.draw_bbox(os.path.join(RESULT_DIR, "00000001.jpg"), first_img, bbox)

    quantized_img = first_img // 32
    car_hist = generate_histogram(quantized_img, bbox)

    for img in images[1:]:
        test_img = cv2.imread(img)
        quantized_img = test_img // 32

        candidate_windows = []
        candidate_scores = []
        max_score = -1
        best_window = None
        search_window = helper.expand_search_window(test_img, bbox,
                                                    search_window_size=60)
        for (x, y, window) in helper.sliding_window(test_img,
                                                    search_window,
                                                    stride=8,
                                                    template_size=(bbox[2], bbox[3])):  # noqa
            tmp_histogram = generate_histogram(quantized_img,
                                               (x, y, bbox[2], bbox[3]),
                                               use_kernel=False)
            ncc_score = helper.ncc(tmp_histogram, car_hist)
            candidate_windows.append((x, y, bbox[2], bbox[3]))
            candidate_scores.append(ncc_score)
            if ncc_score > max_score:
                max_score = ncc_score
                best_window = (x, y, bbox[2], bbox[3])
        bbox = best_window
        helper.draw_bbox(os.path.join(RESULT_DIR, img.split('/')[-1]),
                         test_img, best_window)


if __name__ == "__main__":
    main()
