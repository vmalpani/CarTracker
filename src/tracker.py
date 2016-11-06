import glob

import numpy as np
import cv2

import helper
import selectivesearch


def get_region_proposals(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
    clone = img.copy()
    pruned_regions = []
    for r in regions:
        rect = r['rect']
        if rect[2] == 0 or rect[3] == 0:
            continue
        if ((rect[2] / float(rect[3])) > 0.25 and (rect[2] / float(rect[3])) < 0.75) or ((rect[3] / float(rect[2])) > 0.25 and (rect[3] / float(rect[2])) < 0.75):
            if((rect[2] * float(rect[3])) > 500 and (rect[2] * float(rect[3])) < 50000):
                pruned_regions.append(rect)
                cv2.rectangle(clone, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
    cv2.imshow("sliding window visualizer", clone)


def generate_histogram(img, (x, y, w, h), bins=32, use_kernel=False):
    template = img[y:y+h, x:x+w]
    if use_kernel:
        normalized = template / float(bins)
        kernel = helper.compute_epanechnikov_kernel(h, w)
        normalized[:,:,0] *= kernel
        normalized[:,:,1] *= kernel
        normalized[:,:,2] *= kernel
        template = normalized * float(bins)
    # import pdb
    # pdb.set_trace()
    car_hist = cv2.calcHist([template],[0],None,[bins],[0, bins])
    car_hist = car_hist.astype(float) / np.sum(car_hist)

    return car_hist


def generate_hog_features(img, (x, y, w, h)):
    hog = cv2.HOGDescriptor()
    car_template = img[y:y+h, x:x+w]
    resized_image = cv2.resize(car_template, (64, 128))
    car_features = hog.compute(np.uint8(resized_image))
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
    images = sorted(glob.glob('data/*.jpg'))
    first_img = cv2.imread(images[0])
    bbox = [6, 166, 43, 27]
    helper.draw_bbox("result/00000001.jpg", first_img, bbox)

    # first_img = helper.median_filtering(first_img)
    quantized_img = first_img // 32
    car_hist = generate_histogram(quantized_img, bbox)
    # car_hist = generate_hog_features(first_img, bbox)

    for img in images[1:]:
        test_img = cv2.imread(img)
        # first_img = helper.median_filtering(first_img)
        quantized_img = test_img // 32

        candidate_windows = []
        candidate_scores = []
        max_score = -1
        best_window = None
        search_window = expand_search_window(test_img, bbox,
                                             search_window_size=60)
        for (x, y, window) in helper.sliding_window_new(test_img,
                                                        search_window,
                                                        stride=8,
                                                        template_size=(bbox[2], bbox[3])):  # noqa
            """
            import pdb
            pdb.set_trace()
            normalized_window = window / 255.0
            kernel = helper.compute_epanechnikov_kernel(bbox[3], bbox[2])
            normalized_window[:,:,0] *= kernel
            normalized_window[:,:,1] *= kernel
            normalized_window[:,:,2] *= kernel
            new_window = normalized_window * 255.0
            """
            tmp_histogram = generate_histogram(quantized_img,
                                               (x, y, bbox[2], bbox[3]), use_kernel=False)
            #tmp_histogram = generate_hog_features(test_img,
            #                                   (x, y, bbox[2], bbox[3]))
            ncc_score = helper.ncc(tmp_histogram, car_hist)
            candidate_windows.append((x, y, bbox[2], bbox[3]))
            candidate_scores.append(ncc_score)
            if ncc_score > max_score:
                max_score = ncc_score
                best_window = (x, y, bbox[2], bbox[3])
        bbox = best_window
        helper.draw_bbox("result/" + img.split('/')[-1], test_img, best_window)

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
        cv2.imshow("sliding window visualizer", clone)
        cv2.waitKey(1)
        import time
        time.sleep(0.25)
    helper.draw_bbox("result/0000002.jpg", test_img, best_window)
    cv2.imwrite("result/0000002_crop.jpg", best_crop)
    """

if __name__ == "__main__":
    main()
