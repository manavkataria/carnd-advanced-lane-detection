#!/usr/bin/env ipython
import cv2
import numpy as np
import matplotlib

from utils import imcompare

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class VisionFilters(object):

    def mag_thresh(self, img, sobel_kernel=3, thresh=(0, 255)):
        """
            Define a function to return the magnitude of the gradient
            for a given sobel kernel size and threshold values
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255
        gradmag = (gradmag/scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def dir_threshold(self, img, sobel_kernel=3, thresh=(0, np.pi/2)):
        """
            Define a function to threshold an image for a given range and Sobel kernel """
        # Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Calculate the x and y gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction,
        # apply a threshold, and create a binary image result
        absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(absgraddir)
        binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

        # Return the binary image
        return binary_output

    def abs_sobel_thresh(self, img, orient='x', sobel_kernel=3, thresh=(0, 255)):
        """
            A function that takes an image, gradient orientation,
            and threshold min / max values.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        if orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

        # Create a copy and apply the threshold
        binary_output = np.zeros_like(scaled_sobel)
        # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too

        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return binary_output

    def hls_threshold(self, image, select='h', thresh=(0, 255)):
        # Convert to HLS
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
        H = image[:, :, 0]
        L = image[:, :, 1]
        S = image[:, :, 2]

        if select == 'h':
            channel = H
        elif select == 'l':
            channel = L
        else:
            channel = S

        # Create a copy and apply the threshold
        binary = np.zeros_like(channel)
        binary[(channel >= thresh[0]) & (channel <= thresh[1])] = 1

        return binary


def dstack(binary1, binary2):
    """
        Stack each channel to view their individual contributions in green and blue respectively
        This returns a stack of the two binary images, whose components you can see as different colors
    """
    color_binary = np.dstack((np.zeros_like(binary1), binary1, binary2))
    return color_binary


def hist(img):
    color = ('r', 'g', 'b')
    plt.figure()
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        # cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
        plt.plot(histr, color=col)
        plt.xlim([-1, 3])
    plt.show()


def test_overlay():
    filters = VisionFilters()
    image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')

    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = filters.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # grady = filters.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(10, 200))
    # imcompare(gradx, grady)
    # imcompare(gradx ^ grady, grady - gradx)

    # mag_binary = filters.mag_thresh(image, sobel_kernel=ksize, thresh=(100, 250))
    # imcompare(mag_binary-grady, mag_binary)

    # dir_binary = filters.dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi/2-1.5, np.pi/2-0.5))
    # imcompare(mag_binary, (mag_binary == 1) & (dir_binary==1))

    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # imcompare(image, combined)

    H = filters.hls_threshold(image, select='h', thresh=(70, 100))
    L = filters.hls_threshold(image, select='l', thresh=(170, 255))
    S = filters.hls_threshold(image, select='s', thresh=(100, 255))
    # imcompare(H, L, 'H', 'L')
    # imcompare(gradx, S, 'gradx', 'S')

    gradxs = dstack(gradx, S)
    # hist(gradxs)

    # plt.hist(H)
    # plt.show()
    # debug(np.count_zero(gradxs[:,:,0]), np.count_nonzero(gradxs[:,:,1]), np.count_nonzero(gradxs[:,:,2]))
    imcompare(gradxs, S)


def main():
    pass


if __name__ == '__main__':
    main()
