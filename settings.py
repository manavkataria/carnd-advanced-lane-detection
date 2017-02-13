# settings.py
from numpy import pi

# Debug
DEBUG = True
DISPLAY = True

# Camera Calibration
CAMERA_CALIBRATION_DIR = 'camera_cal'
CAMERA_CALIB_FILE = CAMERA_CALIBRATION_DIR + '/camera_calib.p'
CHESSBOARD_SQUARES = (9, 6)

# Road Test
TEST_IMAGES_DIR = 'test_images'

# Portions of Image to delete from top, bottom, top-width, bottom-width
ROI = {'t': 0.6300, 'b': 0.10, 'tw': 0.15, 'bw': 0.1000}

# Gradient Thresholds
KSIZE = 7

HLS_S_THRESHOLD         = (90, 255)
HLS_H_THRESHOLD         = (0, 60)
SOBEL_GRADX_THRESHOLD   = (20, 100)   # (37, 255)
SOBEL_GRADY_THRESHOLD   = (25, 254)
SOBEL_MAG_THRESHOLD     = (100, 250)  # (110, 255)
SOBEL_DIR_THRESHOLD     = (pi/2-1.5, pi/2-0.5)  #=(0.07, 1.0)   # (0.30, 1.40) Sagar
