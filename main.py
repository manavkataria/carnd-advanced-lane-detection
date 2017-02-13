#!/usr/bin/env ipython
import glob
import matplotlib
import numpy as np
# from moviepy.editor import VideoFileClip

from camera import Camera
from lanes import Lanes
from vision_filters import VisionFilters
from utils import debug, imcompare, dstack, display
from settings import (CAMERA_CALIBRATION_DIR,
                      CHESSBOARD_SQUARES,
                      TEST_IMAGES_DIR,
                      KSIZE,
                      HLS_H_THRESHOLD,
                      HLS_S_THRESHOLD,
                      HLS_L_THRESHOLD,
                      SOBEL_GRADX_THRESHOLD,
                      SOBEL_GRADY_THRESHOLD,
                      SOBEL_MAG_THRESHOLD,
                      SOBEL_DIR_THRESHOLD,
                      GAUSS_KERNEL
                      )

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def test_calibrate_and_transform():
    directory = CAMERA_CALIBRATION_DIR
    filenames = glob.glob(directory + '/*.jpg')
    # filenames = ['camera_cal/calibration2.jpg', 'camera_cal/calibration1.jpg', 'camera_cal/calibration3.jpg']

    camera = Camera(filenames)
    mtx, dist = camera.load_or_calibrate_camera()
    for filename in filenames:
        debug(filename)
        img = mpimg.imread(filename)
        warped, M, Minv = camera.corners_unwarp(img, filename,
                                                CHESSBOARD_SQUARES[0],
                                                CHESSBOARD_SQUARES[1],
                                                mtx, dist)


def filtering_pipeline(image, ksize):
    filters = VisionFilters()

    # S_binary = filters.hls_threshold(image, select='s', thresh=HLS_S_THRESHOLD)
    H_binary = filters.hls_threshold(image, select='h', thresh=HLS_H_THRESHOLD)
    L_binary = filters.hls_threshold(image, select='l', thresh=HLS_L_THRESHOLD)
    # imcompare(S_binary, L_binary, 'S', 'L')
    # hls_sl = dstack(S_binary, L_binary)
    # imcompare(image, hls_sl, None, 'hls_sl')

    # imcompare(L_binary, H_binary, 'L', 'H')
    # hls_lh = dstack(L_binary, H_binary)
    # imcompare(image, hls_lh, None, 'hls_lh')

    # imcompare(S_binary, H_binary, 'S', 'H')
    # hls_sh = dstack(S_binary, H_binary)
    # imcompare(image, hls_sh, None, 'hls_sh')

    gradx = filters.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=SOBEL_GRADX_THRESHOLD)
    grady = filters.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=SOBEL_GRADY_THRESHOLD)
    # gradxy = dstack(gradx, grady)
    # imcompare(image, gradxy, None, 'grad_xy')

    mag_binary = filters.mag_thresh(image, sobel_kernel=ksize, thresh=SOBEL_MAG_THRESHOLD)
    dir_binary = filters.dir_threshold(image, sobel_kernel=ksize, thresh=SOBEL_DIR_THRESHOLD)
    # mag_dir = dstack(mag_binary, dir_binary)
    # imcompare(image, mag_dir, None, 'mag_dir')

    # Combine output from various filters above
    combined = np.zeros_like(image[:, :, 0])
    combined[((L_binary == 1) | (H_binary == 1)) |
             (((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)))] = 1
    # imcompare(image, combined, None, 'All Combined!')

    # Gaussian Blur to smoothen out the noise
    combined = filters.gaussian_blur(combined, GAUSS_KERNEL)

    return combined


def test_road_unwarp():
    directory = TEST_IMAGES_DIR
    filenames = glob.glob(directory + '/*.jpg')
    # filenames = [TEST_IMAGES_DIR + '/test1.jpg',
    #              TEST_IMAGES_DIR + '/test4.jpg',
    #              TEST_IMAGES_DIR + '/test5.jpg',
    #              TEST_IMAGES_DIR + '/signs_vehicles_xygrad.jpg']
    camera = Camera()
    mtx, dist = camera.load_or_calibrate_camera()
    lanes = Lanes(filenames)

    for filename in filenames:
        img = mpimg.imread(filename)
        undistorted = camera.undistort(img, crop=False)
        lane_marked_undistorted = lanes.pipeline(undistorted)
        display(lane_marked_undistorted, filename)


def test_filters():
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
    # test_calibrate_and_transform()
    test_road_unwarp()
    # test_filters()


if __name__ == '__main__':
    main()
