# Advanced-Lane-Detetction-phase-1

## we beleive that diagrams worth thousand words, so this is the high level diagram which represents the pipeline of the project starting from reading a frame to detecting the lane curves and lines.
![pipeline](pipeline.png)

# Description of our pipeline:

## [1] Camera calibration and distortion correction
- We know that when the camera transforms the 3D object points to 2D image points, we will have a distortion in the frames that our camera reads.
- Due to this distortion, we will face problems in some parts of the image which will effect the pipeline process.
- OpenCV provides three functions, namely, **`cv2.findChessboardCorners`**, **`cv2.calibrateCamera`** and **`cv2.undistortto`**.
- We will use a set of chessboard images to calibrate the camera.
### steps:
> * Define a set of object points that represent inside corners in a set of chessboard images.
> * Map the object points to images points by using **`cv2.findChessboardCorners`**.
> * Call **`cv2.calibrateCamera`** with this newly created list of object points and image poins to compute the camera calibration matrix and distortion coefficients.
> * Finally, we can undistort raw images by passing them into **`cv2.undistort`** along with the two params calculated above.



## [2] Perspective Transformation & Region Of Interest Specification
- Now we have an undistorted image, but we also have a lot of unimportant parts of our frame that represent a noise because we are interested in lanes only.
- Since that we have the front-view of the road, we still have these unimportant shapes in our frame (such as trees, other cars ..), So we need to get another view of our frame to focus only on the lanes.
- Imagine that we have an Eye-bird view of our lanes, where we can see the actual view of the lanes "parallel", and we can focus on the lanes only.
- Perspective Transformation which warpes the image into a bird's eye view scene. This makes it easier to detect the lane lines (since they are relatively parallel) and measure their curvature.
### steps:
> * Specify the source coordinates (from front-view) and destination coordinates (from Eye-bird view) to be able to transform from one view to another.
> * Compute the transformation matrix by passing the source and destination points into **`cv2.getPerspectiveTransform`**. These points are determined empirically.
> * Then the undistorted image is warped by passing it into **`cv2.warpPerspective`** along with the transformation matrix.
> * We defined 2 functions: **`get_eye_bird_view()`** which gets the top-view from front-view, and the **`get_front_view()`** which gets back the front-view from top-view 



## [3] Image Thresholding
- the main purpose of this stage is to generate a binary image that contains the pixels that are likely to be a part of the lane lines.
- We have 2 colors for lanes, the yellow and white colors.
- Since that the right lane is almost the white lane, we will focus on the L-Channel from HLS color space to detect the white lanes because we are interested in the lightining property, we will put the minimum threshold to be 80% and maximum threshold to be 100%.
- We will use the H-Channel from HLS color space to detect the yellow lanes, the minimun threshold is 20 and maximum threshold is 30.
> how we got these numbers ? .. the yellow color is focused from the range of 20 degree to 40 degree, we will take the range from 30 deg to less than 60 deg only which is can be calculated using this equation: `angle*(2/3) = threshold`, so 30*(2/3) = 20 and 60*(2/3) = 4  but we will take to the range 30 as our maximum.
- Then we will combine the result from the H-Channel with V-Channel from HSV which repesents the intensity of the yellow color, we will set the minimum threshold = 70 and maximum threshold = 100.
- and finlly we have the left lane and right lane detected successfully.  



## [4] Lane Line Detection : Sliding Window Technique
-  This step involves mapping out the lane lines and determining explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.
-  We first take a histogram along all the columns in the lower half of the image. This involves adding up the pixel values along each column in the image. The two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. These are used as starting points for our search.

From these starting points, we use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.

- The parameters used for the sliding window search are:
> * we will assume that we will have 9 windows per lane 
> * the width of the window = 2 * margin 
> * margin = 100
> * the minimum number of pixels to recenter a single window is minpix , and minpix = 50

- We will fit the values of y by the fit-poly technique by using this equation `x = f(y) = Ay^2 + By + C` since that the x value for multiple points will be almost the same as the Eye-bird (top view) image illustrates that becuase lines are almost vertical and parallel.
- We define a function that will fit and draw the lines using the mentioned fit-poly technique with the mentioned 2nd degree polynomial, this function is named ad `fit_poly(img)`
- the `fit_poly(img)` will do that by calling another method defined which is called `find_lane_pixels(img)`, and this function recieves an image and returns all the non-zero pixels from each window by looping through all windows after calculating the center of each window and the x, y coordinates.
- this is the representation of a single window with all variables we defined in our code:
![window](window.png)
### Curvature of lane:
> * We can compute the radius of curvature at any point x on the lane line represented by the function `x = f(y) = A(y)^2 + By + C` as follows:
> ![Curvature_Rule](curvature_rule(1).png)
> * Since `dx/dy = 2Ay + B` and `d^2x/dy^2 = 2A`
> * Therefore Radius of curvature = ![Curvature_Rule](curvature_rule(2).png)
> * Since the y-values for an image increase from top to bottom, we can compute the lane line curvature at maximum y which is the closest point to the vehicle
### Plotting function:
> This function plot 2 items 
> * the curvature of the lane line and when the vechile move in straight line, we plot curvature = 0 
> * How much vechile is away from the center


## [5] Pipeline
-  This function runs the steps detailed above in sequence to process a video frame by frame 
-  Each frame in the input vedio will pass through sequnce of functions 
    1. undistort function 
    2. get_eye_bird_view function
    3. thresholding function
    4. detect_lane_lines function
    5. get_front_view function
- Then we add (blend) two images the result of the sequence of the functions and the original frame, To put the results on the original frame
 
 
 # To Run the pipeline
 - There are 2 modes (0 or 1) 
     * 0 : the output will be the final video only
     * 1 : the output will be the final video and each step the video will pass on it (Debugging mode)
 - Windows 
    * In the terminal (cmd):
        ``` 
          python main.py 0 INPUT_VIDEO_PATH OUTPUT_VIDEO_PATH
          python main.py 1 INPUT_VIDEO_PATH OUTPUT_VIDEO_PATH
        ```
    * if you want to run in shell script => you must run on Git Bash terminal
        ``` 
          sh projectshell.sh 0 INPUT_VIDEO_PATH OUTPUT_VIDEO_PATH
          sh projectshell.sh 1 INPUT_VIDEO_PATH OUTPUT_VIDEO_PATH
        ```
