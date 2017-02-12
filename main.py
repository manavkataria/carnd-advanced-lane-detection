#!/usr/bin/env ipython
import cv2
import glob
import numpy as np
import matplotlib
import inspect
import pickle
import os

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CAMERA_CALIBRATION_DIR = 'camera_cal'
CAMERA_CALIB_FILE = CAMERA_CALIBRATION_DIR + '/camera_calib.p'
CHESSBOARD_SQUARES = (9, 6)
DISPLAY = False
DEBUG = True


def debug(*args):
    frame, filename, line_number, function_name, lines, index = inspect.stack()[1]
    if DEBUG:
        print('[%s:%d]' % (function_name, line_number), *args)


def display(image, msg='Image', cmap=None):
    if not DISPLAY: return

    if image.ndim == 2:
        cmap = 'gray'

    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(msg, fontsize=30)
    plt.show(block=True)


def imcompare(image1, image2, msg1='Image1', msg2='Image2', cmap1=None, cmap2=None):
    if image1.ndim == 2:
        cmap1 = 'gray'
    if image2.ndim == 2:
        cmap2 = 'gray'

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap=cmap1)
    ax1.set_title(msg1, fontsize=30)
    ax2.imshow(image2, cmap=cmap2)
    ax2.set_title(msg2, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show(block=True)


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
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


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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


# A function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
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


def hls_threshold(image, select='h', thresh=(0, 255)):
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
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
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
    # Read in an image and grayscale it
    image = mpimg.imread('test_images/signs_vehicles_xygrad.jpg')

    # Choose a Sobel kernel size
    ksize = 5  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    # grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(10, 200))
    # imcompare(gradx, grady)
    # imcompare(gradx ^ grady, grady - gradx)

    # mag_binary = mag_thresh(image, sobel_kernel=ksize, thresh=(100, 250))
    # imcompare(mag_binary-grady, mag_binary)

    # dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(np.pi/2-1.5, np.pi/2-0.5))
    # imcompare(mag_binary, (mag_binary == 1) & (dir_binary==1))

    # combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    # imcompare(image, combined)

    H = hls_threshold(image, select='h', thresh=(70, 100))
    L = hls_threshold(image, select='l', thresh=(170, 255))
    S = hls_threshold(image, select='s', thresh=(100, 255))
    # imcompare(H, L, 'H', 'L')
    # imcompare(gradx, S, 'gradx', 'S')

    gradxs = dstack(gradx, S)
    # hist(gradxs)

    # plt.hist(H)
    # plt.show()
    # debug(np.count_zero(gradxs[:,:,0]), np.count_nonzero(gradxs[:,:,1]), np.count_nonzero(gradxs[:,:,2]))
    imcompare(gradxs, S)


def accumalate_objpoints_and_imagepoints(filenames):
    # 3D Object Points in Real World Space
    objpoints = []
    # 2D Points in Image Space
    imgpoints = []

    for filename in filenames:
        img = mpimg.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SQUARES)
        debug(filename, ret, len(corners) if corners is not None else 0)

        if ret is True:
            img = cv2.drawChessboardCorners(img, CHESSBOARD_SQUARES, corners, ret)
            display(img, 'corners %s %s' % (filename, str(CHESSBOARD_SQUARES)))

        # objp are contained within objpoints
        objp = np.zeros((CHESSBOARD_SQUARES[0] * CHESSBOARD_SQUARES[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHESSBOARD_SQUARES[0], 0:CHESSBOARD_SQUARES[1]].T.reshape(-1, 2)  # x, y coordinates 0,0 .. 8,4

        if ret is True:
            imgpoints.append(corners)
            objpoints.append(objp)
        else:
            debug("Could Not Find All (54) Chessboard Corners for:", filename)

    debug('Found %d/%d chessboard corners' % (len(objpoints), len(filenames)))
    return objpoints, imgpoints, img.shape[:2]


def undistort(img, mtx, dist, crop=True):
    if crop:
        # Regular Undistort Image; Cropped
        dst = cv2.undistort(img, mtx, dist, None, mtx)
    else:
        # Uncropped
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    display(dst)
    return dst


def calibrate_camera(filenames):
    if (os.path.isfile(CAMERA_CALIB_FILE)):
        print("File found:" + CAMERA_CALIB_FILE)
        print("Loading: camera calib params")
        mtx, dist = pickle.load(open(CAMERA_CALIB_FILE, "rb"))
        return mtx, dist

    mtx, dist = None, None
    opts, ipts, shape = accumalate_objpoints_and_imagepoints(filenames)
    if len(opts) > 0 and len(ipts) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, shape, None, None)

    # Save camera calibration params
    with open(CAMERA_CALIB_FILE, 'wb') as f:
        debug("Saving calib params: ", CAMERA_CALIB_FILE)
        pickle.dump([mtx, dist], f)

    return mtx, dist


def corners_unwarp(img, filename, nx, ny, mtx, dist):
    undistorted_img = undistort(img, mtx, dist)

    if img.ndim == 3:
        gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
        debug("Grayed Shape", gray.shape)
    else:
        gray = undistorted_img

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SQUARES)

    warped = np.zeros_like(img)
    M = None
    # import ipdb; ipdb.set_trace()
    # If found, draw corners
    if ret is True:
        cv2.drawChessboardCorners(undistorted_img, (nx, ny), corners, ret)
        display(undistorted_img)

        src = np.float32([corners[0][0],
                          corners[nx-1][0],
                          corners[(nx)*(ny-1)][0],
                          corners[(nx)*(ny)-1][0]])

        # Output Image Size
        img_size = gray.shape[1], gray.shape[0]
        x = img_size[0]/nx
        y = img_size[1]/ny

        debug(img_size, x, y)

        # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
        dst = np.float32([[x/2, y/2],
                          [img_size[0]-x/2, y/2],
                          [x/2, img_size[1]-y/2],
                          [img_size[0]-x/2, img_size[1]-y/2]])

        # d) use cv2.getPerspectiveTransform() to get M, the transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # e) use cv2.warpPerspective() to warp your image to a top-down view
        warped = cv2.warpPerspective(undistorted_img, M, img_size, flags=cv2.INTER_LINEAR)
        imcompare(undistorted_img, warped, 'undist_' + filename[-6:], 'warped_' + filename[-6:])
        debug("Warped Shape", filename, warped.shape)

    return warped, M, Minv


def test_camera_calibration(filenames):
    mtx, dist = calibrate_camera(filenames)


def test_calibrate_and_transform(filenames):
    # Test
    global DEBUG
    DEBUG = True
    mtx, dist = calibrate_camera(filenames)

    DEBUG = True
    for filename in filenames:
        debug(filename)
        img = mpimg.imread(filename)
        warped, M, Minv = corners_unwarp(img, filename,
                                         CHESSBOARD_SQUARES[0],
                                         CHESSBOARD_SQUARES[1],
                                         mtx, dist)


def main():
    # Test
    directory = CAMERA_CALIBRATION_DIR
    filenames = glob.glob(directory + '/*')
    # filenames = ['camera_cal/calibration2.jpg', 'camera_cal/calibration1.jpg', 'camera_cal/calibration3.jpg']
    test_calibrate_and_transform(filenames)


if __name__ == '__main__':
    main()
