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

# Portions of Image to delete from top, bottom, top-width, bottom-width
ROI = {'t': 0.6300, 'b': 0.10, 'tw': 0.15, 'bw': 0.1000}
