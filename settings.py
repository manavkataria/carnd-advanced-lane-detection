# settings.py
from numpy import pi

# Debug
DEBUG = True
DISPLAY = True

# Video Input
VIDEO_INPUT = 'project_video.mp4'
OUTPUT_DIR = 'output_images/'
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
GAUSS_KERNEL = 31

# HLS_S_THRESHOLD         = (90, 255)     # Both White and Yellow
HLS_S_THRESHOLD         = (180, 255)    # Yellow picked up and a little bit of white  (UNUSED)
# HLS_S_THRESHOLD         = (90, 120)     # White picked up not yellow; Too Tight
# HLS_S_THRESHOLD         = (90, 180)   # Full Noise
HLS_H_THRESHOLD         = (0, 45)       # Clean Yellow; not white
HLS_L_THRESHOLD         = (245, 255)    # White Only; Clean!
SOBEL_GRADX_THRESHOLD   = (20, 100)   # (37, 255)
SOBEL_GRADY_THRESHOLD   = (25, 254)
SOBEL_MAG_THRESHOLD     = (100, 250)  # (110, 255)
SOBEL_DIR_THRESHOLD     = (pi/2-1.5, pi/2-0.5)  #=(0.07, 1.0)   # (0.30, 1.40) Sagar
