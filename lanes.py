#!/usr/bin/env ipython
import cv2
import numpy as np
import matplotlib

from utils import warper
from settings import ROI

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.image as mpimg


class Lanes(object):

    def __init__(self, filenames):
        self.shape = None
        self.roi = None
        self.filenames = filenames
        self.init_shape()
        self.init_roi()

    def init_shape(self):
        img = mpimg.imread(self.filenames[0])
        self.shape = img.shape
        self.img_width = self.shape[1]
        self.img_height = self.shape[0]

    def init_roi(self):
        IMAGE_WIDTH = self.img_width
        IMAGE_HEIGHT = self.img_height
        x1ROI, y1ROI = int(IMAGE_WIDTH/2 * (1 - ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
        x2ROI, y2ROI = int(IMAGE_WIDTH/2 * (1 + ROI['tw'])), int(IMAGE_HEIGHT * ROI['t'])
        x3ROI, y3ROI = int(IMAGE_WIDTH   * (1 - ROI['bw'])), int(IMAGE_HEIGHT * (1 - ROI['b']))
        x4ROI, y4ROI = int(IMAGE_WIDTH   * ROI['bw']),       int(IMAGE_HEIGHT * (1 - ROI['b']))
        self.roi = [(x1ROI, y1ROI), (x2ROI, y2ROI), (x3ROI, y3ROI), (x4ROI, y4ROI)]
        return self.roi

    def overlay_roi(self, image):
        if self.roi is None:
            raise Exception('Error: How the heck is ROI none! Did you forget to initialize?!')
        [(x1ROI, y1ROI), (x2ROI, y2ROI), (x3ROI, y3ROI), (x4ROI, y4ROI)] = self.roi
        warp_zero = np.zeros_like(image).astype(np.uint8)
        pts = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)
        cv2.fillPoly(warp_zero, [pts], (0, 255, 255))
        image = cv2.addWeighted(image, 1, warp_zero, 0.3, 0)
        return image

    def crop_to_region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def perspective_transform(self, img):
        if self.roi is None:
            raise Exception('Error: How the heck is ROI none! Did you forget to initialize?!')
        [(x1ROI, y1ROI), (x2ROI, y2ROI), (x3ROI, y3ROI), (x4ROI, y4ROI)] = self.roi

        # Source ROI Trapeziod
        src = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)

        # Input image masked with ROI trapezoid
        img_ROI = self.crop_to_region_of_interest(img, [src])

        # Destination ROI rectangle
        dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
        dst_full = np.array([[0, 0], [self.img_width, 0], [self.img_width, self.img_height], [0, self.img_height]], np.float32)

        src = np.float32(src)
        # Perspective transform to get bird's eye view
        warped, M, Minv = warper(img_ROI, src, dst)
        warped_full, Mfull, Minv_full = warper(img_ROI, src, dst_full)
        # TODO(manav): self.Minv = -> global Minv, Minv_full ?

        return warped, warped_full
