#!/usr/bin/env ipython
import cv2
import numpy as np
import matplotlib

from utils import warper
from settings import ROI

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Lanes(object):

    def __init__(self, filenames):
        self.shape = None
        self.roi = None
        self.filenames = filenames

        # Perspective Transformation Matrices
        self.M_cropped = None
        self.Minv_cropped = None
        self.M_scaled = None
        self.Minv_scaled = None

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

        if self.M_cropped is None or self.M_scaled is None:
            if self.roi is None:
                raise Exception('Error: How the heck is ROI none! Did you forget to initialize?!')
            [(x1ROI, y1ROI), (x2ROI, y2ROI), (x3ROI, y3ROI), (x4ROI, y4ROI)] = self.roi

            # Source ROI Trapeziod
            src = np.array([[x1ROI, y1ROI], [x2ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.int32)

            # Input image masked with ROI trapezoid
            img_ROI = self.crop_to_region_of_interest(img, [src])

            # Destination ROI rectangle
            dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
            dst_scaled = np.array([[0, 0], [self.img_width, 0], [self.img_width, self.img_height], [0, self.img_height]], np.float32)

            src = np.float32(src)
            # Perspective transform to get bird's eye view
            warped_cropped, self.M_cropped, self.Minv_cropped = warper(img_ROI, src, dst)
            warped_scaled, self.M_scaled, self.Minv_scaled = warper(img_ROI, src, dst_scaled)
        else:
            warped_cropped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
            warped_scaled = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

        return warped_cropped, warped_scaled

    def fit_lane_lines(self, binary_warped, visualize=True):
        """ Udacity Code 👎 😠 """

        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        # plt.plot(left_fitx, ploty, color='yellow')
        # plt.plot(right_fitx, ploty, color='yellow')
        # plt.xlim(0, binary_warped.shape[1])
        # plt.ylim(binary_warped.shape[0], 0)
        # plt.imshow(out_img)

        return ploty, left_fitx, right_fitx

    def overlay_and_unwarp(self, image, ploty, left_fitx, right_fitx, invWarp=True):
        # Create an image to draw the lines on
        color_warp =  np.zeros((image.shape[1], image.shape[0], image.shape[2]), np.uint8)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        pts = np.squeeze(pts)
        pts = pts.astype(int)

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, [pts], (0, 255, 0))

        if invWarp:
            # Warp the blank image back to original perspective space
            color_warp = cv2.warpPerspective(color_warp, self.Minv_scaled, (image.shape[1], image.shape[0]))

        # Combine the result with the original image
        overlayed = cv2.addWeighted(image, 1, color_warp, 0.3, 0)

        return overlayed
