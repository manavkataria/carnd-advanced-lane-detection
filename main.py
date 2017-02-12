#!/usr/bin/env ipython
import glob
import matplotlib

from camera import Camera
from lanes import Lanes
from vision_filters import VisionFilters
from utils import debug, imcompare, dstack, hist
from settings import (CAMERA_CALIBRATION_DIR,
                      CHESSBOARD_SQUARES,
                      TEST_IMAGES_DIR)

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

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


def test_road_unwarp():
    directory = TEST_IMAGES_DIR
    filenames = glob.glob(directory + '/*.jpg')

    camera = Camera()
    mtx, dist = camera.load_or_calibrate_camera()

    lanes = Lanes(filenames)

    for filename in filenames:
        img = mpimg.imread(filename)
        roi_cropped = lanes.overlay_roi(img)
        undistorted_img = camera.undistort(roi_cropped, crop=False)
        # imcompare(roi_cropped, undistorted_img, filename, 'undistorted')
        roi_perspective, roi_perspective_full = lanes.perspective_transform(roi_cropped)
        imcompare(undistorted_img, roi_perspective_full, filename, 'Perspective')
        # TODO(Manav): Gradients


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
    # test_calibrate_and_transform()
    # test_road_unwarp()
    test_overlay()


if __name__ == '__main__':
    main()
