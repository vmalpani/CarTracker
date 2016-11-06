#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import time
import multiprocessing
import glob

from PIL import Image
import selectivesearch
import cv2
import caffe

import helper
from tracker import expand_search_window

PRETRAINED = 'model/cifar10_quick_iter_4000.caffemodel.h5'
MODEL_FILE = 'model/cifar10_quick.prototxt'
LABELS = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck',
    ]
CHUNK_SIZE = 30


def load_image(image_fh, color=True):
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


def is_overlap((left1, top1, width1, height1), (left2, top2, width2, height2)):
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

    # iou = intersection/float(union)
    if intersect_width > 0 and intersect_height > 0:
        return True
    else:
        return False


def generate_region_proposals(img):
    (img_lbl, regions) = selectivesearch.selective_search(img, scale=500,
                                                          sigma=0.9,
                                                          min_size=100)
    return regions


def prun_region_proposals(region, search_window, check_overlap):
    rect = region['rect']
    size = region['size']

    if rect[2] == 0 or rect[3] == 0:
        return False

    if check_overlap:
    	if not is_overlap(rect, search_window):
            return False

    if rect[2] / float(rect[3]) > 0.25 and \
       rect[2] / float(rect[3]) < 0.75 or \
       rect[3] / float(rect[2]) > 0.25 and \
       rect[3] / float(rect[2]) < 0.75:
        if size > 500 and size < 50000:
            return True
        else:
            return False
    else:
        return False


class Predictions:

    def __init__(self):
        self.cifar10 = self._generate_net()
        self.transformer = self._generate_transformer()

    def _generate_net(self):
        # mean = np.array([104, 117, 123])
        # net = caffe.Classifier(MODEL_FILE, PRETRAINED,
        # mean=mean, channel_swap=(2,1,0),
        # raw_scale=255, image_dims=(256, 256))

        net = caffe.Net(MODEL_FILE, weights=PRETRAINED,
                        phase=caffe.TEST)
        caffe.set_mode_cpu()
        return net

    def _generate_transformer(self):
        transformer = \
            caffe.io.Transformer({'data': self.cifar10.blobs['data'].data.shape})  # noqa
        # move image channels to outermost dimension
        transformer.set_transpose('data', (2, 0, 1))
        # subtract the dataset-mean value in each channel
        transformer.set_mean('data', np.array([104, 117, 123]))
        # rescale from [0, 1] to [0, 255]
        # transformer.set_raw_scale('data', 255)
        # swap channels from RGB to BGR
        transformer.set_channel_swap('data', (2, 1, 0))
        return transformer

    def generate_predictions(self, img, search_window):
        regions = generate_region_proposals(img)

        # p = multiprocessing.Pool(multiprocessing.cpu_count())
        # mask = p.map(prun_region_proposals, (regions, search_window))
        # p.close()
        # p.join()
        mask = [prun_region_proposals(r, search_window, True) for r in regions]

        pruned_regions = np.array(regions)[np.array(mask)]
        # clone = img.copy()
        # cv2.rectangle(clone, (search_window[0], search_window[1]),
        #               (search_window[0] + search_window[2],
        #               search_window[1] + search_window[3]), (255, 0,
        #               0), 2)
        # for region in pruned_regions:
        #     r = region['rect']
        #     cv2.rectangle(clone, (r[0], r[1]), (r[0] + r[2], r[1] + r[3]), \
        #     	                 (0, 255, 0), 2)
        # cv2.imshow("test", clone)
        # import pdb
        # pdb.set_trace()
        # cv2.destroyAllWindows()

        inputs = np.zeros((CHUNK_SIZE, 3, 32, 32), dtype=np.float32)

        max_prob = 0
        best_bbox = None
        rects = []
        count = 0
        for region in pruned_regions:
            count += 1
            bbox = region['rect']
            rects.append(bbox)
            crop_img = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0]
                           + bbox[2]]

            # crop_img = cv2.resize(crop_img, (32, 32))
            # test_data = np.swapaxes(np.asarray([crop_img]),3,1)
            # test_data = np.swapaxes(test_data,2,3)

            inputs[count - 1] = self.transformer.preprocess('data', crop_img)
            if count == CHUNK_SIZE:
                out = self.cifar10.forward_all(data=inputs)
                # out['prob'][:, 1] += out['prob'][:, 9]
                max_idx = out['prob'][:, 1].argmax()
                prob_automobile = out['prob'][:, 1][max_idx]
                if prob_automobile > max_prob:
                    max_prob = prob_automobile
                    best_bbox = rects[max_idx]
                inputs = np.zeros((CHUNK_SIZE, 3, 32, 32),
                                  dtype=np.float32)
                rects = []
                count = 0

        # forward pass on the last batch

        if len(rects) > 0:
            out = self.cifar10.forward_all(data=inputs[:len(rects)])
            # out['prob'][:, 1] += out['prob'][:, 9]
            max_idx = out['prob'][:, 1].argmax()
            prob_automobile = out['prob'][:, 1][max_idx]
            if prob_automobile > max_prob:
                max_prob = prob_automobile
                best_bbox = rects[max_idx]

        return (best_bbox, max_prob)


if __name__ == '__main__':
    pred = Predictions()
    images = sorted(glob.glob('data/*.jpg'))
    first_img = cv2.imread(images[0])
    bbox = [6, 166, 43, 27]
    search_window_size = 90
    search_window = expand_search_window(first_img, bbox,
                                         search_window_size)
    for fname in images[1:]:
        img = cv2.imread(fname)
        if img is not None:
            t1 = time.time()
            (best_bbox, max_prob) = pred.generate_predictions(img, search_window)
            print best_bbox, max_prob
            if best_bbox:
                print time.time() - t1
                clone = img.copy()
                cv2.rectangle(clone, (best_bbox[0], best_bbox[1]),
                              (best_bbox[0] + best_bbox[2],
                              best_bbox[1] + best_bbox[3]), (255, 0,
                              0), 5)
                cv2.rectangle(clone, (search_window[0], search_window[1]),
                              (search_window[0] + search_window[2],
                              search_window[1] + search_window[3]), (0, 255,
                              0), 2)
                helper.draw_bbox('result_detection/' + fname.split('/')[-1],
                                 clone, best_bbox)
                search_window = expand_search_window(img, best_bbox,
                                                     search_window_size)
