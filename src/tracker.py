import numpy as np
import cv2

import helper


def get_template_histogram(img, (x, y, w, h)):
    car_template = img[y:y+h, x:x+w]


def main():
    input_image = cv2.imread('data/00000001.jpg')
    bbox = [6, 166, 43, 27]
    helper.draw_bbox("result/00000001.jpg", input_image, bbox)
    get_template_histogram(input_image, bbox)

if __name__ == "__main__":
    main()
