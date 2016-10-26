import numpy as np
import cv2

import helper


def generate_histogram(quantized_img, (x, y, w, h), bins=16):
    template = quantized_img[y:y+h, x:x+w]
    car_hist, _ = np.histogram(template.flatten(), bins=bins, range=(0, bins))
    car_hist = car_hist.astype(float) / np.sum(car_hist)
    return car_hist


def generate_hog_features(img, (x, y, w, h)):
    hog = cv2.HOGDescriptor()
    car_template = img[y:y+h, x:x+w]
    car_features = hog.compute(np.uint8(car_template))
    return car_features


def expand_search_window(img, rect, search_window_size=120):
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


def main():
    first_img = cv2.imread('data/00000001.jpg')
    bbox = [6, 166, 43, 27]
    helper.draw_bbox("result/00000001.jpg", first_img, bbox)

    # first_img = helper.median_filtering(first_img)
    quantized_img = first_img // 16
    car_hist = generate_histogram(quantized_img, bbox)

    test_img = cv2.imread('data/00000002.jpg')
    # test_img = helper.median_filtering(test_img)
    quantized_img = test_img // 16
    # test_hist = generate_histogram(quantized_img, (19, 169, 46, 25))

    # 6.00,193.00,6.00,166.00,49.00,166.00,49.00,193.00
    # 19.00,194.00,19.00,169.00,65.00,169.00,65.00,194.00
    # x, y, w, h = 19, 169, 46, 25

    candidate_windows = []
    candidate_scores = []
    max_score = -1
    best_window = None
    best_crop = None
    cv2.imshow("test", test_img)
    cv2.waitKey(5)
    search_window = expand_search_window(test_img, bbox,
                                         search_window_size=120)
    for (x, y, window) in helper.sliding_window_new(test_img, search_window,
                                                    stride=8,
                                                    template_size=(bbox[2], bbox[3])):  # noqa
        tmp_histogram = generate_histogram(quantized_img,
                                           (x, y, bbox[2], bbox[3]))
        ncc_score = helper.chi_squared(tmp_histogram, car_hist)
        candidate_windows.append((x, y, bbox[2], bbox[3]))
        candidate_scores.append(ncc_score)
        if ncc_score > max_score:
            max_score = ncc_score
            best_window = (x, y, bbox[2], bbox[3])
            best_crop = window

        """
        # uncomment to visualize prev bbox, expanded search box, sliding window
        clone = test_img.copy()
        cv2.rectangle(clone, (bbox[0], bbox[1]),
                             (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                             (255, 0, 0), 2)
        cv2.rectangle(clone, (search_window[0], search_window[1]),
                             (search_window[0] + search_window[2],
                              search_window[1] + search_window[3]),
                             (0, 0, 255), 2)
        cv2.rectangle(clone, (x, y), (x + bbox[2], y + bbox[3]),
                                     (0, 255, 0), 2)
        cv2.imshow("window", clone)
        cv2.waitKey(1)
        import time
        time.sleep(0.25)
        """
    helper.draw_bbox("result/0000002.jpg", test_img, best_window)
    cv2.imwrite("result/0000002_crop.jpg", best_crop)

if __name__ == "__main__":
    main()
