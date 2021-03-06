#!/usr/bin/env ipython
import cv2
import numpy as np
import matplotlib

from utils import warper, imcompare, debug
from settings import ROI, KSIZE

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Lanes(object):

    def __init__(self, filenames=None, undistort=None, filtering_pipeline=None, shape=(720, 1280)):
        self.shape = None
        self.roi = None
        self.filenames = filenames

        # Camera Undistort
        self.undistort = undistort

        # Filtering pipeline:
        self.filtering_pipeline = filtering_pipeline

        # Perspective Transformation Matrices
        self.M_cropped = None
        self.Minv_cropped = None
        self.M_scaled = None
        self.Minv_scaled = None

        self.init_shape(shape)
        self.init_roi()

        # Debugging
        self.save = None
        self.count = 0

        # Scale from pixels to meter per pixel scale
        self.y_m_per_pix =  30/720 # meters per pixel in y dimension
        self.x_m_per_pix = 3.7/700 # meters per pixel in x dimension

        # Approx Camera Placement Offset in meter
        self.lane_offset_bias = -1.5

    def init_shape(self, shape):
        if self.filenames:
            img = mpimg.imread(self.filenames[0])
            self.shape = img.shape
        else:
            self.shape = shape

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
    # TODO(Manav): Tweaking Improvements
    # Make Parallel Lane Lines

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
            warped_cropped = cv2.warpPerspective(img, self.M_cropped, (img.shape[0], img.shape[1]), flags=cv2.INTER_LINEAR)
            warped_scaled = cv2.warpPerspective(img, self.M_scaled, (img.shape[0], img.shape[1]), flags=cv2.INTER_LINEAR)

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


    def fill_lane_poly(self, image, left_fitx, ploty, mid_fitx, color):
        """ Note this is a mutating function """
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_left_mid = np.array([np.flipud(np.transpose(np.vstack([mid_fitx, ploty])))])
        left_to_mid_pts = np.hstack((pts_left, pts_left_mid))
        left_to_mid_pts = np.squeeze(left_to_mid_pts)
        left_to_mid_pts = left_to_mid_pts.astype(int)
        cv2.fillPoly(image, [left_to_mid_pts], color)

    def fill_lane_polys(self, image, left_fitx, ploty, right_fitx, left_color, mid_color, right_color):
        """ Note this is a mutating function """
        # Identify/Highlight car's offset position in lane
        ratio = (float(image.shape[1])/2 - left_fitx[-1]) / (right_fitx[-1] - left_fitx[-1])
        mid_fitx = (left_fitx + right_fitx)*0.5
        car_fitx = (left_fitx + right_fitx)*ratio

        if ratio <= 0.5:
            self.fill_lane_poly(image, left_fitx, ploty, car_fitx, left_color)
            self.fill_lane_poly(image, car_fitx, ploty, mid_fitx, mid_color)
            self.fill_lane_poly(image, mid_fitx, ploty, right_fitx, right_color)
        else:
            self.fill_lane_poly(image, left_fitx, ploty, mid_fitx, left_color)
            self.fill_lane_poly(image, mid_fitx, ploty, car_fitx, mid_color)
            self.fill_lane_poly(image, car_fitx, ploty, right_fitx, right_color)

    def overlay_and_unwarp(self, image, ploty, left_fitx, right_fitx, invWarp=True):
        # Create an image to draw the lines on
        color_warp =  np.zeros((image.shape[1], image.shape[0], image.shape[2]), np.uint8)

        # Recast the x and y points into a polygon for cv2.fillPoly()
        left_color = (0, 100, 0)
        mid_color = (255, 0, 0)
        right_color = (0, 180, 0)
        self.fill_lane_polys(color_warp, left_fitx, ploty, right_fitx,
                             left_color, mid_color, right_color)

        # Draw a Image Center Line
        # midline_bottom_pt = (int(image.shape[0]/2), image.shape[1]-1)
        # midline_top_pt = (int(image.shape[0]/2), image.shape[1]-250)
        # cv2.arrowedLine(color_warp, midline_bottom_pt, midline_top_pt, (0,0,0), 20)

        if invWarp:
            # Warp the blank image back to original perspective space
            color_warp = cv2.warpPerspective(color_warp, self.Minv_scaled, (image.shape[1], image.shape[0]))

        # Combine the result with the original image
        overlayed = cv2.addWeighted(image, 1, color_warp, 0.3, 0)

        return overlayed

    def calculate_curvature(self, ploty, left_fitx, right_fitx):

        # Fit the lane markings on coordinates
        left_fit_cr = np.polyfit(ploty * self.y_m_per_pix, left_fitx * self.x_m_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * self.y_m_per_pix, right_fitx * self.x_m_per_pix, 2)
        y_eval_m = np.max(ploty) * self.y_m_per_pix

        # Calculate radii of curvature
        left_curve_radius = ((1. + (2*left_fit_cr[0]*y_eval_m +
                                    left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curve_radius = ((1. + (2*right_fit_cr[0]*y_eval_m +
                                     right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Compute centre of lane markings and centre of vehicle
        off_centre_pixels = ((self.img_width/2) - (left_fitx[-1] + right_fitx[-1])/2)
        off_centre_m = off_centre_pixels * self.x_m_per_pix + self.lane_offset_bias

        return left_curve_radius, right_curve_radius, off_centre_m

    def put_metrics_on_image(self, image, left_curve_radius, right_curve_radius, off_center_m):
        """ Note this is a mutating function """

        cv2.putText(image, 'Radius of Lanes: %0.1f(m); %0.1f(m)' % (left_curve_radius,
                    right_curve_radius), (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(image, 'Position from Centre: %0.1f(m)' % (off_center_m), (100, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4, cv2.LINE_AA)


    def pipeline(self, image):
        # Skip Frames for faster debugging
        # if self.count <= 600:
        #     self.count += 1
        #     return image

        undistorted = self.undistort(image, crop=False)
        imcompare(image, undistorted, 'Original', 'Undistorted')
        roi_overlayed = self.overlay_roi(undistorted)
        imcompare(undistorted, roi_overlayed, 'Undistorted', 'ROI Mask Overlayed')

        cropped_perspective, scaled_perspective = self.perspective_transform(roi_overlayed)
        imcompare(roi_overlayed, scaled_perspective, 'ROI Mask Overlayed', 'Scaled Perspective')

        # Color and Gradient Filters + Denoising
        filtered = self.filtering_pipeline(scaled_perspective, ksize=KSIZE)
        imcompare(scaled_perspective, filtered, 'Scaled Perspective', 'Vision Filter Pipeline')

        try:
            ploty, left_fitx, right_fitx = self.fit_lane_lines(filtered)
            self.save = (ploty, left_fitx, right_fitx)
        except:
            mpimg.imsave('hard/%d.jpg' % self.count, image)
            debug('Error: Issue at Frame %d' % self.count)
            (ploty, left_fitx, right_fitx) = self.save

        self.count += 1
        lane_marked_undistorted = self.overlay_and_unwarp(undistorted, ploty, left_fitx, right_fitx)

        (left_curve_radius,
         right_curve_radius,
         off_centre_m) = self.calculate_curvature(ploty, left_fitx, right_fitx)

        self.put_metrics_on_image(lane_marked_undistorted,
                                  left_curve_radius,
                                  right_curve_radius,
                                  off_centre_m)

        return lane_marked_undistorted
