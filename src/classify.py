import caffe
import numpy as np
import cv2
import selectivesearch
import time
import multiprocessing

import helper


PRETRAINED = '/Users/vmalpani/code/caffe/examples/cifar10/cifar10_quick_iter_4000.caffemodel.h5'
MODEL_FILE = '/Users/vmalpani/code/caffe/examples/cifar10/cifar10_quick.prototxt'


def _prun_region_proposals(region):
    rect = region['rect']
    size = region['size']
    if rect[2] == 0 or rect[3] == 0:
    	return False
    if ((rect[2] / float(rect[3])) > 0.25 and (rect[2] / float(rect[3])) < 0.75) or \
       ((rect[3] / float(rect[2])) > 0.25 and (rect[3] / float(rect[2])) < 0.75):
        if(size > 500 and size < 50000):
        	return True
        else:
        	return False
    else:
		return False


class Predictions():
	def __init(self):
		self.cifar10 = self._generate_net()
		self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
					   'ship', 'truck']

	def _generate_net(self):
		mean = np.array([104, 117, 123])
		net = caffe.Classifier(MODEL_FILE, PRETRAINED, 
							   mean=mean, channel_swap=(2,1,0), 
							   raw_scale=255, image_dims=(256, 256))
		caffe.set_mode_cpu()
		return net

	def _generate_region_proposals(self, img):
		img_lbl, regions = selectivesearch.selective_search(img, scale=500, sigma=0.9, min_size=10)
		return regions

	def _prun_region_proposals_old(self, regions):
		pruned_rects = []
		clone = img.copy()
		for r in regions:
		    rect = r['rect']
		    if rect[2] == 0 or rect[3] == 0:
		        continue
		    if ((rect[2] / float(rect[3])) > 0.25 and (rect[2] / float(rect[3])) < 0.75) or ((rect[3] / float(rect[2])) > 0.25 and (rect[3] / float(rect[2])) < 0.75):
		        if((rect[2] * float(rect[3])) > 500 and (rect[2] * float(rect[3])) < 50000):
		            pruned_rects.append(rect)
		            cv2.rectangle(clone, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
		cv2.imshow("sliding window visualizer", clone)
		return pruned_rects

	def generate_predictions(self, test_img):
		t1 = time.time()
		self.cifar10 = self._generate_net()
		t2 = time.time()
		regions = self._generate_region_proposals(img)
		t3 = time.time()


		p = multiprocessing.Pool(multiprocessing.cpu_count())
		mask = p.map(_prun_region_proposals, regions)
		p.close()
		p.join()
		pruned_regions = np.array(regions)[np.array(mask)]
		t4 = time.time()

		max_prob = 0
		best_bbox = None
		for region in pruned_regions:
			bbox = region['rect']
			crop_img = img[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
			crop_img = cv2.resize(crop_img, (32, 32))
			test_data = np.swapaxes(np.asarray([crop_img]),3,1)
			test_data = np.swapaxes(test_data,2,3)
			out = self.cifar10.forward_all(data=test_data)
			prob = out['prob'][0]
			prob_automobile = prob[1] + prob[9]
			if prob_automobile > max_prob:
				max_prob = prob_automobile
				best_bbox = bbox
		t5 = time.time()
		print t5 - t4
		print t4 - t3
		print t3 - t3
		print t2 - t1
		return best_bbox, max_prob

if __name__ == "__main__":
	img = cv2.imread('00000112.jpg')
	pred = Predictions()
	
	clone = img.copy()
	t1 = time.time()
	best_bbox, max_prob = pred.generate_predictions(img)
	print "\n\n\n"
	print time.time() - t1
	cv2.rectangle(clone, (best_bbox[0], best_bbox[1]), (best_bbox[0] + best_bbox[2], 
		                  best_bbox[1] + best_bbox[3]), (255, 0, 0), 2)
	helper.draw_bbox("./test_output.jpg", clone, best_bbox)
	cv2.imshow("sliding window", clone)
	time.sleep(5)
	cv2.destroyAllWindows()
	print best_bbox, max_prob
