# settings.py

# Debug
DEBUG = True
DISPLAY = True

# Camera Calibration
CAMERA_CALIBRATION_DIR = 'camera_cal'
CAMERA_CALIB_FILE = CAMERA_CALIBRATION_DIR + '/camera_calib.p'
CHESSBOARD_SQUARES = (9, 6)

# Road Test
TEST_IMAGES_DIR = 'test_images'

# Image ROI Crop Percentage from [left, top, right, bottom]
ROI_bbox = [0.0, 0.40, 0.0, 0.13]

# Portions of Image to delete from top, bottom, top-width, bottom-width
ROI = {'t': 0.6300, 'b': 0.0700, 'tw': 0.1710, 'bw': 0.0000}
