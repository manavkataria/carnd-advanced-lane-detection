#!/usr/bin/env ipython
import cv2
import glob
import numpy as np
import matplotlib
import pickle
import os

from utils import debug, display, imcompare, warper
from settings import (CAMERA_CALIBRATION_DIR,
                      CAMERA_CALIB_FILE,
                      CHESSBOARD_SQUARES,
                      TEST_IMAGES_DIR,
                      ROI)

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Camera(object):

    def __init__(self, chessboard_images_filepath=None, camera_calib_filepath=CAMERA_CALIB_FILE):
        self.chessboard_images_filepath = chessboard_images_filepath
        self.camera_calib_filepath = camera_calib_filepath
        self.mtx = None
        self.dist = None

    def accumalate_objpoints_and_imagepoints(self, filenames):
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
                # display(img, 'corners %s %s' % (filename, str(CHESSBOARD_SQUARES)))

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

    def undistort(self, img, mtx=None, dist=None, crop=True):
        if (mtx is None or dist is None):
            if (self.mtx is not None and self.dist is not None):
                mtx, dist = self.mtx, self.dist
            else:
                raise Exception("Error: Invalid Request! Matrix or Distortion Coeff Missing")

        if crop:
            # Regular Undistort Image; Cropped
            dst = cv2.undistort(img, mtx, dist, None, mtx)
        else:
            # Uncropped
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # display(dst)
        return dst

    def load_or_calibrate_camera(self):
        if (os.path.isfile(self.camera_calib_filepath)):
            debug("File found:" + self.camera_calib_filepath)
            debug("Loading: camera calib params")
            mtx, dist = pickle.load(open(self.camera_calib_filepath, "rb"))
            self.mtx, self.dist = mtx, dist
            return mtx, dist

        if self.chessboard_images_filepath is not None:
            filenames = self.chessboard_images_filepath
        else:
            raise Exception("Error: Conflict! No Calibration Parameters Found!")

        mtx, dist = None, None
        opts, ipts, shape = self.accumalate_objpoints_and_imagepoints(filenames)
        if len(opts) > 0 and len(ipts) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(opts, ipts, shape, None, None)

        # Save camera calibration params
        with open(self.camera_calib_filepath, 'wb') as f:
            debug("Saving calib params: ", self.camera_calib_filepath)
            pickle.dump([mtx, dist], f)

        self.mtx, self.dist = mtx, dist
        return mtx, dist

    def corners_unwarp(self, img, filename, nx, ny, mtx=None, dist=None):
        if (mtx is None or dist is None):
            if (self.mtx is not None and self.dist is not None):
                mtx, dist = self.mtx, self.dist
            else:
                raise Exception("Error: Invalid Request! Matrix or Distortion Coeff Missing")

        undistorted_img = self.undistort(img, mtx, dist)

        if img.ndim == 3:
            gray = cv2.cvtColor(undistorted_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = undistorted_img

        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SQUARES)

        warped = np.zeros_like(img)
        M, Minv = None, None

        # If found, draw corners
        if ret is True:
            cv2.drawChessboardCorners(undistorted_img, (nx, ny), corners, ret)
            # display(undistorted_img)

            src = np.float32([corners[0][0],
                              corners[nx-1][0],
                              corners[(nx)*(ny-1)][0],
                              corners[(nx)*(ny)-1][0]])

            # Output Image Size
            img_size = gray.shape[1], gray.shape[0]
            x = img_size[0]/nx
            y = img_size[1]/ny

            # c) define 4 destination points dst = np.float32([[,],[,],[,],[,]])
            dst = np.float32([[x/2, y/2],
                              [img_size[0]-x/2, y/2],
                              [x/2, img_size[1]-y/2],
                              [img_size[0]-x/2, img_size[1]-y/2]])

            warped, M, Minv = warper(undistorted_img, src, dst)
            imcompare(undistorted_img, warped, 'undist_' + filename[-6:], 'warped_' + filename[-6:])

        return warped, M, Minv


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

    def region_of_interest(self, img, vertices):
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
        img_ROI = self.region_of_interest(img, [src])

        # Destination ROI rectangle
        dst = np.array([[x4ROI, y1ROI], [x3ROI, y2ROI], [x3ROI, y3ROI], [x4ROI, y4ROI]], np.float32)
        dst_full = np.array([[0, 0], [self.img_width, 0], [self.img_width, self.img_height], [0, self.img_height]], np.float32)

        src = np.float32(src)
        # Perspective transform to get bird's eye view
        warped, M, Minv = warper(img_ROI, src, dst)
        warped_full, Mfull, Minv_full = warper(img_ROI, src, dst_full)
        # TODO(manav): self.Minv = -> global Minv, Minv_full ?

        return warped, warped_full


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


def main():
    # test_calibrate_and_transform()
    test_road_unwarp()


if __name__ == '__main__':
    main()
