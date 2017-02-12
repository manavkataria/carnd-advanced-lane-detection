#!/usr/bin/env ipython
import cv2
import glob
import numpy as np
import matplotlib
import pickle
import os

from utils import debug, display, imcompare
from settings import (CAMERA_CALIBRATION_DIR,
                      CAMERA_CALIB_FILE,
                      CHESSBOARD_SQUARES,
                      TEST_IMAGES_DIR)

matplotlib.use('TkAgg')  # MacOSX Compatibility
matplotlib.interactive(True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Camera(object):

    def __init__(self, chessboard_images_filepath, camera_calib_filepath=CAMERA_CALIB_FILE):
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

    def undistort(self, img, mtx, dist, crop=True):
        mtx = self.mtx
        dist = self.dist

        if crop:
            # Regular Undistort Image; Cropped
            dst = cv2.undistort(img, mtx, dist, None, mtx)
        else:
            # Uncropped
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        display(dst)
        return dst

    def calibrate_camera(self):
        if (os.path.isfile(self.camera_calib_filepath)):
            debug("File found:" + self.camera_calib_filepath)
            debug("Loading: camera calib params")
            mtx, dist = pickle.load(open(self.camera_calib_filepath, "rb"))
            self.mtx, self.dist = mtx, dist
            return mtx, dist

        filenames = self.chessboard_images_filepath

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
        if mtx is None or dist is None:
            mtx, dist = self.mtx, self.dist

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
            display(undistorted_img)

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

            warped, M, Minv = self.warper(undistorted_img, src, dst)
            imcompare(undistorted_img, warped, 'undist_' + filename[-6:], 'warped_' + filename[-6:])

        return warped, M, Minv

    def warper(self, img, src, dst):
        # Compute and apply perpective transform
        img_size = (img.shape[1], img.shape[0])
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        # keep same size as input image
        warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)
        return warped, M, Minv


def test_calibrate_and_transform():
    directory = CAMERA_CALIBRATION_DIR
    filenames = glob.glob(directory + '/*.jpg')
    # filenames = ['camera_cal/calibration2.jpg', 'camera_cal/calibration1.jpg', 'camera_cal/calibration3.jpg']

    camera = Camera(filenames)
    mtx, dist = camera.calibrate_camera()

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
    mtx, dist = camera.calibrate_camera(filenames)

def main():
    test_calibrate_and_transform()

if __name__ == '__main__':
    main()
