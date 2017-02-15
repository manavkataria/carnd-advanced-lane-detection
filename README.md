Table of Contents
=================
   1. [Advanced Lane Lines](#advanced-lane-lines)
      * [Pipeline](#pipeline)
      * [Pipeline Images](#pipeline-images)
   1. [Files](#files)
      * [Usage](#usage)
   1. [Challenges](#challenges)
      * [Lack of Intuition](#lack-of-intuition)
      * [Building Intuition with Visual Augmentation](#building-intuition-with-visual-augmentation)
      * [Road Textures](#road-textures)
   1. [Shortcomings &amp; Future Enhancements](#shortcomings--future-enhancements)
      * [Enhancements for future](#enhancements-for-future)
   1. [Acknowledgements &amp; References](#acknowledgements--references)

---

# Advanced Lane Lines
This video contains results and illustration of challenges encountered during this project:

[![youtube thumb](https://cloud.githubusercontent.com/assets/2206789/22967459/670853b4-f31b-11e6-9eef-1493e728e7f9.jpg)](https://youtu.be/6lf099n2LkI)

---

## Pipeline
1. Camera Calibration
   * RGB2Gray using `cv2.cvtColor`
   * [Finding](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/camera.py#L36) and [Drawing](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/camera.py#L40) Corners using `cv2.findChessboardCorners` and `cv2.drawChessboardCorners`
   * Identifying Camera Matrix and Distortion Coefficients using `cv2.calibrateCamera`
   * [Undistort](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/camera.py#L56-L73)
     * Cropped using `cv2.undistort`
     * Uncropped additionally using `cv2.getOptimalNewCameraMatrix`
   * Perspective Transform in `corners_unwarp`
2. Filters using [`filtering_pipeline`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L49-L86)
   * RGB to HSL
   * H & L Color Threshold Filters
   * Gradient, Magnitude and Direction Filters
   * Careful Combination of the above
   * Guassian Blur to eliminate noise `K=31`
3. Lane Detection [`pipeline`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L294)
   * [`undistort`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/camera.py#L56-L73)
   * [`perspective_transform`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L96-L123)
       * `crop_to_region_of_interest`
   * [`filtering_pipeline`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L49-L86)
   * [`fit_lane_lines`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L125)
       * `left_fitx`
       * `right_fitx`
   using [`histogram[:midpoint]`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L136) and [`sliding window`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L160) to capture points forming a lane line along with 2nd Order Polynomial curve fitting, identifies
   * [`overlay_and_unwarp`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L238)
       * Car's Trajectory `car_fitx`
       * Lane Center `mid_fitx`
       * `fill_lane_polys`
   * [`calculate_curvature`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L263)
       * `left_curve_radius`
       * `right_curve_radius`
       * `off_centre_m`
   * [`put_metrics_on_image`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L285)
   * finally returning an _undistorted_ image
4. [Save as Video.mp4  ](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L118)

## Pipeline Images

### Camera Calibration > Undistort
![screen shot 2017-02-15 at 3 23 28 am](https://cloud.githubusercontent.com/assets/2206789/22973046/3aeea448-f331-11e6-875b-e9f65db7a591.jpg)
### Camera Calibration > Perspective Transform
![screen shot 2017-02-15 at 3 24 07 am](https://cloud.githubusercontent.com/assets/2206789/22973069/557e48e0-f331-11e6-94f0-2e2668481503.jpg)

### Lane Detection > Undistort
![screen shot 2017-02-15 at 3 35 36 am](https://cloud.githubusercontent.com/assets/2206789/22973024/27513360-f331-11e6-809c-9af79a24d477.jpg)
### Lane Detection > ROI Mask Overlay
![screen shot 2017-02-15 at 3 36 01 am](https://cloud.githubusercontent.com/assets/2206789/22973023/274cea58-f331-11e6-9e1a-958faffd200b.jpg)
### Lane Detection > Perspective Transform
![screen shot 2017-02-15 at 3 36 19 am](https://cloud.githubusercontent.com/assets/2206789/22973022/274ba5d0-f331-11e6-9ac1-4032fac69ff7.jpg)
### Lane Detection > Filtering Pipeline
![screen shot 2017-02-15 at 3 36 35 am](https://cloud.githubusercontent.com/assets/2206789/22973021/274b031e-f331-11e6-8b95-88b172f3f00e.jpg)
### Lane Detection > Offset and Curvature Identified
![screen shot 2017-02-15 at 3 37 06 am](https://cloud.githubusercontent.com/assets/2206789/22973025/27523f94-f331-11e6-95ee-95d173fc15d9.jpg)
Minor: Note the position from center is represented as a positive 0.2(m). Compare with images below.

# Files
The project was designed to be modular and reusable. The significant independent domains get their own `Class` and an individual file:
  1. `camera.py` - Camera Calibration
  1. `lanes.py` - Lane Detection
  1. `main.py` - Main test runner with [`test_road_unwarp`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L89), [`test_calibrate_and_transform`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L33-L46) and [`test_filters`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/main.py#L123)
  1. `utils.py` - Handy utils like [`imcompare`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32),  [`warper`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L32), [`debug`](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/utils.py#L14-L17) shared across modules
  1. `settings.py` - Settings shared across module
  1. `vision_filters.py` - Gradient, magnitude, direction, Sobel filters and related
  1. `README.md` - description of the development process (this file)

All files contain **detailed comments** to explain how the code works. Refer Udacity Repository [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines/) - for calibration images, test images and test videos

## Usage
Repository includes all required files and can be used to rerun advanced lane line detection on a given video. Set configuration values in `settings.py` and run the `main.py` python script.
```
$ grep 'INPUT\|OUTPUT' -Hn settings.py
settings.py:9:      INPUT_VIDEOFILE = 'project_video.mp4'
settings.py:11:     OUTPUT_DIR = 'output_images/'

$ python main.py
[load_or_calibrate_camera:77] File found: camera_cal/camera_calib.p
[load_or_calibrate_camera:78] Loading: camera calib params
[test_road_unwarp:112] Processing Video:  project_video.mp4
[MoviePy] >>>> Building video output_images/project_video_output.mp4
[MoviePy] Writing video output_images/project_video_output.mp4  
100%|██████████████████████████████████████████████████████████████▉| 1260/1261 [11:28<00:00,  1.99it/s]
[MoviePy] Done.
[MoviePy] >>>> Video ready: output_images/project_video_output.mp4

$ open output_images/project_video_output.mp4
```

# Challenges

### Lack of Intuition
There was no visual indication for developing an intuition to identify Car's Lane-Center offset by looking at the video. For example in this image the car is quit off the center towards the left side of the lane. But it doesn't show:  

**Figure: Frame with No Intuitive Indication of Off-Center Distance**
![initial fillpoly no off-center indication](https://cloud.githubusercontent.com/assets/2206789/22967458/67078fba-f31b-11e6-8f37-8e34cab24ff7.jpg)

### Building Intuition with Visual Augmentation

To get an intuitive feel, I decided to identify, approximate and visualize the position of the car in the lane with respect to lane-center. In order to achieve this, I identified four different lane lines:
  1. Left Lane Marker Line
  1.  Approximate Car's Trajectory Line
  1.  Lane Center Line, and
  1.  Right Lane Marker Line

**Figure: Frame with Intuitive Off-Center Highlights**
![working sample2](https://cloud.githubusercontent.com/assets/2206789/22967455/6706de80-f31b-11e6-815a-07a1d65ecf10.jpg)

## Road Textures

**Figure: Frame with Bridge**
![bridge](https://cloud.githubusercontent.com/assets/2206789/22967456/67071206-f31b-11e6-8281-927c0f30c2e7.jpg)

**Figure: Frame with Shadows (Sidenote: Smaller Off-Center Position)**
![shadows note the off-center indicator](https://cloud.githubusercontent.com/assets/2206789/22967457/67072368-f31b-11e6-9062-cf0cb01236d0.jpg)


A simple and elegant solution to approximate the trajectory of the car was to use the existing lane end markers and interpolate between them. Using the center of the image as car's position and identifying its relative position between the lane ends, I came up with an approximate car trajectory, as follows:
```
# Identify/Highlight car's offset position in lane
ratio = (float(image.shape[1])/2 - left_fitx[-1]) / (right_fitx[-1] - left_fitx[-1])
mid_fitx = (left_fitx + right_fitx)*0.5
car_fitx = (left_fitx + right_fitx)*ratio
```
Then it was just a matter of coloring them with a generic `fill_lane_polys()`[lanes.py#L222-L236](https://github.com/manavkataria/carnd-advanced-lane-detection/blob/master/lanes.py#L222-L236) function to draw polygons on lanes given a lane fit line. The off-center distance was thus displayed the lane itself in RED.

```
self.fill_lane_poly(image, car_fitx, ploty, mid_fitx, mid_color)
```

# Shortcomings & Future Enhancements

**Figure: Example of a frame where the current implementation falls apart**
![falls apart](https://cloud.githubusercontent.com/assets/2206789/22967460/670947d8-f31b-11e6-9faf-91435321946f.jpg)

## Enhancements for future
1. Use ∞ for straight lanes beyond say ~10km radius
1. Add Intermediate Processing Frames to Video
1. Smoothen Radius Metric using Moving Average (lowpass filter)
    * Use a `Line` class to keep track of left and right lane lines
    * Consider Weighted Averaging (on line-length, for example)
1. Reuse lane markers to eliminate full frame search for subsequent frames
1. Use sliders to tune thresholds (Original idea courtesy **[Sagar Bhokre](https://github.com/sagarbhokre)**)
1. Work on harder challenge videos

# Acknowledgements & References
* **[Sagar Bhokre](https://github.com/sagarbhokre)** - for project skeleton & constant support
* **[Caleb Kirksey](https://github.com/ckirksey3)** - for motivation and the idea of using camera bias
* [CarND-Advanced-Lane-Lines](https://github.com/udacity/CarND-Advanced-Lane-Lines/) - Udacity Repository containing calibration images, test images and test videos
