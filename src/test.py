import unittest
import helper as hp
import classify as cl
import numpy as np
import cv2


class HelperTests(unittest.TestCase):

    def test_ncc(self):
        x = [201, 31, 171, 68, 146, 170, 220, 35, 62, 188, 44, 51, 63, 88,
             132, 68, 129, 265, 102, 64, 47, 215, 220, 115, 175, 104, 80,
             365, 68, 286, 35, 62]
        y = [632, 50, 221, 223, 30, 94, 154, 58, 98, 130, 33, 160, 76, 212,
             422, 77, 98, 196, 175, 87, 174, 117, 105, 78, 113, 63, 43, 47,
             56, 29, 77, 90]
        self.assertEqual(round(hp.ncc(x, y), 4), 0.1474)

        x = [202, 27, 162, 71, 151, 175, 153, 35, 65, 200, 40, 56, 59, 83,
             70, 123, 175, 280, 251, 94, 51, 65, 210, 104, 173, 92, 368, 82,
             78, 278, 34, 63]
        self.assertEqual(round(hp.ncc(x, y), 4), 0.0970)

    def test_compute_epanechnikov_kernel(self):
        expected_kernel = [[0.9918499228395061, 0.9992283950617284,
                            0.9918499228395061, 0.9228395061728395],
                           [0.9918499228395061, 0.9992283950617284,
                            0.9918499228395061, 0.9228395061728395],
                           [0.90234375, 0.9375, 0.90234375, 0.75]]
        actual_kernel = hp.compute_epanechnikov_kernel(3, 4).tolist()
        np.testing.assert_almost_equal(actual_kernel,
                                       expected_kernel)

    def test_expand_search_window(self):
        expected_window = [15, 15, 120, 120]
        actual_window = list(hp.expand_search_window(np.zeros((200, 200)),
                                                     (50, 50, 50, 50)))
        self.assertEqual(actual_window, expected_window)

    def test_sliding_window(self):
        expected_window_coords = [[50, 50], [75, 50], [50, 75], [75, 75]]
        actual_window_coords = []
        for item in list(hp.sliding_window(np.zeros((200, 200)),
                                           (50, 50, 50, 50), 25,
                                           (10, 10))):
            actual_window_coords.append([item[0], item[1]])
        self.assertEqual(actual_window_coords, expected_window_coords)


class ClassifyTests(unittest.TestCase):

    def test_generate_predictions(self):
        pred = cl.Predictions()
        first_img = cv2.imread('data/00000001.jpg')
        expected_pred = [0, 0, 133, 63]
        actual_pred = list(pred.generate_predictions(first_img,
                                                     [0, 0,
                                                      first_img.shape[0],
                                                      first_img.shape[1]])[0])
        self.assertEqual(actual_pred, expected_pred)

    def test_is_overlap(self):
        rect1 = (11, 11, 24, 24)
        rect2 = (10, 11, 20, 20)
        self.assertTrue(cl._is_overlap(rect1, rect2))

        rect3 = (40, 42, 20, 20)
        self.assertFalse(cl._is_overlap(rect1, rect3))

    def test_prun_image_regions(self):
        region = {'rect': (1, 1, 1, 1), 'size': 500000}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 1, 1), 'size': 5}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 1, 0), 'size': 5000}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 0, 1), 'size': 5000}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 100000, 1), 'size': 5000}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 1, 100000), 'size': 5000}
        self.assertFalse(cl._prun_region_proposals((0, 0, 100, 100), region))

        region = {'rect': (1, 1, 5, 10), 'size': 5000}
        self.assertTrue(cl._prun_region_proposals((0, 0, 100, 100), region))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
