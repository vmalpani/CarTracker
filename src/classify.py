import numpy as np
import time
import multiprocessing
import glob
import os
from functools import partial

import selectivesearch
import cv2
import caffe

import helper


PRETRAINED = 'model/cifar10_full_iter_70000.caffemodel.h5'
MODEL_FILE = 'model/cifar10_quick.prototxt'
LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck']
RESULT_DIR = 'result_detection'
CHUNK_SIZE = 30
SEARCH_WINDOW_SIZE = 90


def _is_overlap((left1, top1, width1, height1),
                (left2, top2, width2, height2)):
    """checks for overlap and intersection over union of two bounding boxes"""
    right1 = left1 + width1
    bottom1 = top1 + height1

    right2 = left2 + width2
    bottom2 = top2 + height2

    intersect_width = max(0, min(right1, right2) - max(left1, left2))
    intersect_height = max(0, min(bottom1, bottom2) - max(top1, top2))

    # intersection = intersect_width * intersect_height

    # area1 = width1 * height1
    # area2 = width2 * height2
    # union = area1 + area2 - intersection

    if intersect_width > 0 and intersect_height > 0:
        return True
    else:
        return False


def _generate_region_proposals(img):
    """generate region proposals"""
    _, regions = selectivesearch.selective_search(img, scale=500,
                                                  sigma=0.9,
                                                  min_size=100)
    return regions


def _prun_region_proposals(search_window, region):
    """select valid region proposals"""
    rect = region['rect']
    size = region['size']

    # check if width or height is 0
    if rect[2] == 0 or rect[3] == 0:
        return False

    # check if there is any overlap
    if not _is_overlap(rect, search_window):
        return False

    # check for valid aspect ratio of vehicles
    # as the camera position is fixed
    if rect[2] / float(rect[3]) > 0.25 and \
       rect[2] / float(rect[3]) < 0.75 or \
       rect[3] / float(rect[2]) > 0.25 and \
       rect[3] / float(rect[2]) < 0.75:
        if size > 500 and size < 40000:
            return True
        else:
            return False
    else:
        return False


class Predictions:
    """Makes predictions on region proposals using caffe cifar10 model"""

    def __init__(self):
        self.cifar10 = self._generate_net()
        self.transformer = self._generate_transformer()

    def _generate_net(self):
        """instantiate a caffe net"""
        net = caffe.Net(MODEL_FILE, weights=PRETRAINED,
                        phase=caffe.TEST)
        caffe.set_mode_cpu()
        return net

    def _generate_transformer(self):
        """instantiate a input data transformer for caffe net"""
        transformer = caffe.io.Transformer({'data': self.cifar10.blobs['data'].data.shape})  # noqa

        # move image channels to outermost dimension
        transformer.set_transpose('data', (2, 0, 1))

        # subtract the dataset-mean value in each channel
        transformer.set_mean('data', np.array([104, 117, 123]))

        # rescale from [0, 1] to [0, 255]
        # transformer.set_raw_scale('data', 255)

        # swap channels from RGB to BGR
        transformer.set_channel_swap('data', (2, 1, 0))

        return transformer

    def _do_forward_pass(self, inputs, rects, max_prob, best_bbox):
        """does a forward pass on inputs and returns the best bbox"""
        out = self.cifar10.forward_all(data=inputs)

        # add class probabilities of automobiles and trucks
        # out['prob'][:, 1] += out['prob'][:, 9]

        max_idx = out['prob'][:, 1].argsort()[::-1][0]
        prob_automobile = out['prob'][:, 1][max_idx]
        if prob_automobile > max_prob:
            max_prob = prob_automobile
            best_bbox = rects[max_idx]

        return max_prob, best_bbox

    def generate_predictions(self, img, search_window):
        """
        img: image read into a numpy array
        search_window: (x, y, w, h) of search window to prun region proposals
        best_bbox: return best bounding box
        max_prob: returns corresponding probability of best bounding box

        Takes in an image and search window, generates region proposals, pruns
        them, does forward passes on the pruned regions, returns the best one
        """

        # generate region proposals over the whole image
        regions = _generate_region_proposals(img)

        # prun the regions using some heuristics
        p = multiprocessing.Pool(multiprocessing.cpu_count())
        func = partial(_prun_region_proposals, search_window)
        mask = p.map(func, regions)
        p.close()
        p.join()

        pruned_regions = np.array(regions)[np.array(mask)]

        # initilize an array with input batch size for forward pass
        inputs = np.zeros((CHUNK_SIZE, 3, 32, 32), dtype=np.float32)
        rects = []
        count = 0

        max_prob = 0
        best_bbox = None
        for region in pruned_regions:
            count += 1
            bbox = region['rect']
            rects.append(bbox)
            crop_img = img[bbox[1]:bbox[1] + bbox[3],
                           bbox[0]:bbox[0] + bbox[2]]

            inputs[count - 1] = self.transformer.preprocess('data', crop_img)
            if count == CHUNK_SIZE:
                # extract the best bounding box prediction
                max_prob, best_bbox = self._do_forward_pass(inputs,
                                                            rects,
                                                            max_prob,
                                                            best_bbox)
                # clear variables for next batch
                inputs = np.zeros((CHUNK_SIZE, 3, 32, 32), dtype=np.float32)
                rects = []
                count = 0

        # forward pass on the last batch
        if len(rects) > 0:
            max_prob, best_bbox = self._do_forward_pass(inputs[:len(rects)],
                                                        rects,
                                                        max_prob,
                                                        best_bbox)

        return (best_bbox, max_prob)


def main():
    """main driver function"""

    # create results directory if not already
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    pred = Predictions()
    images = sorted(glob.glob('data/*.jpg'))

    # take first image and ground truth bbox
    first_img = cv2.imread(images[0])
    helper.draw_bbox(os.path.join(RESULT_DIR, "00000001.jpg"), first_img,
                     [[6, 166, 43, 27]])
    # defines search window for the next frame
    # based on car's location in current frame
    # [6, 166, 43, 27] - ground truth location of car in first frame
    search_window = helper.expand_search_window(first_img, [6, 166, 43, 27],
                                                SEARCH_WINDOW_SIZE)
    for fname in images[1:]:
        img = cv2.imread(fname)
        if img is not None:
            t1 = time.time()
            # takes about 2-2.5s per image using cpu mode on macbook pro
            (best_bbox, max_prob) = pred.generate_predictions(img, search_window)  # noqa
            # print best_bbox, max_prob
            # print time.time() - t1
            if best_bbox:
                # if valid bbox found, draw and write to disk
                helper.draw_bbox(os.path.join(RESULT_DIR, fname.split('/')[-1]),  # noqa
                                 img, [best_bbox])
                # define search window for next frame
                search_window = helper.expand_search_window(img, best_bbox,
                                                            SEARCH_WINDOW_SIZE)


if __name__ == '__main__':
    main()
