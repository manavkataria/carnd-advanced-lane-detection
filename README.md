Table of Contents
=================

   * Advanced Lane Findin
      * Some heading [TODO]
---

# Advanced Lane Lines
Features:
* Camera Calibration 
* Color Transformation: RGB to HSL
* Perspective Transform
* Offset from Lane Center
* Radius of Lane Curvature

## TLDR; Watch the Video
This video contains subtitles mentioning salient points and challenges encountered during the project.

[![Youtube Video](https://cloud.githubusercontent.com/assets/2206789/22684201/316bb47c-ecd0-11e6-92c7-66eb5790d286.jpg)](https://goo.gl/zhD2jV)

---
# Files
My project includes the following files:
* `camera.py` - Camera Calibration
* `lanes.py` - Lane Detection
* `main.py` - Main test runner
* `utils.py` - Utils shared across module
* `settings.py` - Settings shared across module
* `vision_filters.py` - Gradient, magnitude, direction, Sobel filters and related
* `README.md` - description of the development process (this file)
* [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines/) - Udacity Repository containing calibration images, test images and test videos

All files contain **detailed comments** to explain how the code works.

## How to run the program
Repository includes all required files and can be used to rerun advanced lane line detection on a given video. 
```
$ cat settings.py | grep 'INPUT\|'
settings.py:9:      INPUT_VIDEOFILE = 'project_video.mp4'
settings.py:11:     OUTPUT_DIR = 'output_images/'

$ python main.py
[TODO]

$ open output_images/project_video_output.mp4
```

Here's a link to the output video `project_video_output.mp4` (same as above) [TODO]. 

# Acknowledgements & References
* **[Sagar Bhokre](https://github.com/sagarbhokre)** - for project skeleton and constant support
* **[Caleb Kirksey](https://github.com/ckirksey3)** - for his motivating company during the course of this project
