import unittest
import helper
import numpy as np

class HelperTests(unittest.TestCase):
    def test_ncc(self):
        x = [201, 31, 171, 68, 146, 170, 220, 35, 62, 188, 44, 51, 63, 88, 132, 68, 129, 265, 102,
             64, 47, 215, 220, 115, 175, 104, 80, 365, 68, 286, 35, 62]
        y = [632, 50, 221, 223, 30, 94, 154, 58, 98, 130, 33, 160, 76, 212, 422, 77, 98, 196, 175,
             87, 174, 117, 105, 78, 113, 63, 43, 47, 56, 29, 77, 90]
        self.assertEqual(round(ncc(x, y), 4), 0.1474)

        x = [202, 27, 162, 71, 151, 175, 153, 35, 65, 200, 40, 56, 59, 83, 70, 123, 175, 280, 251,
             94, 51, 65, 210, 104, 173, 92, 368, 82, 78, 278, 34, 63]
        self.assertEqual(round(ncc(x, y), 4) == 0.0970)

    def test_compute_epanechnikov_kernel(self):
        desired_kernel = np.array([[0.99184992, 0.9992284, 0.99184992, 0.92283951]
                                  [0.99184992, 0.9992284, 0.99184992, 0.92283951]
                                  [0.90234375, 0.9375, 0.90234375, 0.75]])
        np.testing.assert_almost_equal(compute_epanechnikov_kernel(3, 4), desired_kernel)
