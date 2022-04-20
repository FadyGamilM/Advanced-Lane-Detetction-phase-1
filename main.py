# IMPORT ESSENTIAL LIBRARIES
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from docopt import docopt
from IPython.display import HTML, Video
from moviepy.editor import VideoFileClip
import glob
import sys

#! [1] CALIBRATE THE CAMERA AND REMOVE DISTORTION
objpoints = []
imgpoints = []

# 9 intersections in x-axis and 6 intersections in y-axis
nx = 9
ny = 6

def calibrate_camera():
  """ Calibrate the camera using list of calibration images and return the 
      camera matrix and distortion coeff    
   """
  # create matrix with size of all corners with 3 dim
  objp = np.zeros((nx*ny, 3), np.float32)
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

  # Make a list of calibration images
  frames = glob.glob('./camera_cal/calibration*.jpg')

  # loop through the frames paths
  for frame in frames:
    # read the frame using the given path
    # img = mpimg.imread(frame)
    calibration_img_BGR = cv2.imread(frame)
    calibration_img_RGB = cv2.cvtColor(calibration_img_BGR, cv2.COLOR_BGR2RGB)
    # convert to gray scale
    gray = cv2.cvtColor(calibration_img_BGR, cv2.COLOR_BGR2GRAY)
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(calibration_img_RGB, (nx, ny))
    if ret:
      imgpoints.append(corners)
      objpoints.append(objp)

  shape = (calibration_img_RGB.shape[1], calibration_img_RGB.shape[0])
  ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)
  return mtx, dist

camera_Matrix, distortion_Coeff = calibrate_camera() 

def undistort(test_frame):
  """ 
    # Method number 1 in pipeline of the main function ...
    this method remove the distortion from the image
  """
  return cv2.undistort(test_frame, camera_Matrix, distortion_Coeff, None, camera_Matrix)


#! [2] FINDING TOP 'BIRD-EYE' VIEW FROM GIVEN FRONT VIEW
src = np.float32([(550, 460),     # top-left
                  (150, 720),     # bottom-left
                  (1200, 720),    # bottom-right
                  (770, 460)])    # top-right
dst = np.float32([(100, 0),
                  (100, 720),
                  (1100, 720),
                  (1100, 0)])

def get_transform_matrix(src, dst):
  """ get the transform matrix from src and dst dimensions """
  M = cv2.getPerspectiveTransform(src, dst)
  return M

def get_inverse_transform_matrix(src, dst):
  """ get the inverse transform matrix from src and dst dimensions """
  M_inv = cv2.getPerspectiveTransform(dst, src)
  return M_inv

# get the transform and inverse transform matrix
transform_matrix = get_transform_matrix(src, dst)
inverse_transform_matrix = get_inverse_transform_matrix(src, dst)

def get_eye_bird_view(img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
  """ 
  Take a front view image and transform to top view
  Parameters:
      img (np.array): A front view image
      img_size (tuple): Size of the image (width, height)
      flags : flag to use in cv2.warpPerspective()

  Returns:
      Image (np.array): Top view image
  """
  return cv2.warpPerspective(img, transform_matrix, img_size, flags=flags)
  
def get_front_view(img, img_size=(1280, 720), flags=cv2.INTER_LINEAR):
  """ Take a top view image and transform it to front view

  Parameters:
      img (np.array): A top view image
      img_size (tuple): Size of the image (width, height)
      flags (int): flag to use in cv2.warpPerspective()

  Returns:
      Image (np.array): Front view image
  """
  return cv2.warpPerspective(img, inverse_transform_matrix, img_size, flags=flags)

#! [3] THRESHOLDING TO DETECT THE LANES FROM THE IMAGE
def threshold_rel(img, lo, hi):
  vmin = np.min(img)
  # print("vmin = ", vmin)
  vmax = np.max(img)
  # print("vmax = ", vmax)
  vlo = vmin + (vmax - vmin) * lo
  # print("vlo = ", vlo)
  vhi = vmin + (vmax - vmin) * hi
  # print("vhi = ", vhi)
  # print(np.uint8((img >= vlo) & (img <= vhi)) * 255)
  return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
  # print(np.uint8((img >= lo) & (img <= hi)) * 255)
  return np.uint8((img >= lo) & (img <= hi)) * 255

def thresholding(img):
  """ Take an image and extract all relavant pixels.

  Parameters:
      img (np.array): Input image

  Returns:
      binary (np.array): A binary image represent all positions of relavant pixels.
  """
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  h_channel = hls[:,:,0]
  l_channel = hls[:,:,1]
  s_channel = hls[:,:,2]
  v_channel = hsv[:,:,2]
  # we choosed the L-channel to detect the right lane because the right lane is almost white
  # so we put the min threshold to 80% and max threshold to be 100%
  right_lane = threshold_rel(l_channel, 0.8, 1.0)
  right_lane[:,:750] = 0
  # print("right_lane", right_lane)
  # choose the range from 30*(2/3) = 20 and 60*(2/3) = 40 but we will take the range from 20 to 30 only because the dark yellow is not in our case
  left_lane = threshold_abs(h_channel, 20, 30)
  # the value of the color is from 70 % to 100 % in intensity
  left_lane &= threshold_rel(v_channel, 0.7, 1.0)
  left_lane[:,550:] = 0
  img2 = left_lane | right_lane
  return img2

def detect_lane_lines(img):
  """
  this method wrap the whole lane_lines_detection process by calling these 2 methods
  """
  # extract more features from the image .. 
  extract_features(img)
  # try to fit and draw the lane lines using polyfit technique
  return fit_poly(img)

def get_histogram(img):
  """ function that returns a histogram for a passed image """ 
  # take the bottom half of the image by taking all rows starting 
  # from the middle of the image to the bottom of the image
  # and take all the width
  bottom_half = img[img.shape[0]//2:,:]
  # return the sum of all the rows starting from the bottom half of the image
  return np.sum(bottom_half, axis=0)

# REQUIRED VARIABLES
nwindows = 9
margin = 100
minpix = 50
binary = None
left_fit = None
right_fit = None
dir = []
nonzero = None
nonzerox = None
nonzeroy = None
clear_visibility = True
window_height = 0

def extract_features(img):
  global nonzero
  global nonzerox
  global nonzeroy
  global window_height
  window_height = np.int(img.shape[0]//nwindows)
  # Identify the x and y positions of all nonzero pixel in the image
  nonzero = img.nonzero()
  # returns the nonzero from width (cols)
  nonzerox = np.array(nonzero[1])
  # returns the nonzero from height (rows)
  nonzeroy = np.array(nonzero[0])

def pixels_in_window(center, margin, height):
  global nonzerox
  global nonzeroy
  """ Return all pixel that in a specific window

  Parameters:
      center (tuple): coordinate of the center of the window
      margin (int): half width of the window
      height (int): height of the window

  Returns:
      pixelx (np.array): x coordinates of pixels that lie inside the window
      pixely (np.array): y coordinates of pixels that lie inside the window
  """
  topleft = (center[0]-margin, center[1]-height//2)
  bottomright = (center[0]+margin, center[1]+height//2)

  condx = (topleft[0] <= nonzerox) & (nonzerox <= bottomright[0])
  condy = (topleft[1] <= nonzeroy) & (nonzeroy <= bottomright[1])
  return nonzerox[condx&condy], nonzeroy[condx&condy]

def find_lane_pixels(img):
  """
  this function find the x,y coordinates of all pixels in left lane
  and also find the x,y coordinates of all pixels in right lane
  returns:
    > leftx
    > rightx
    > lefty
    >righty
  """
  global window_height
  global nwindows
  global margin
  global minpix

  assert(len(img.shape) == 2)
  # Create an output image to draw on and visualize the result
  out_img = np.dstack((img, img, img))

  # get the histogram of the image
  histogram = get_histogram(img)
  # get the mid point of this histogram
  midpoint = histogram.shape[0]//2
  # divide the histogram to 2 parts 
  # -> getting the max index of max point from starting point of the histogram to the midpoint
  leftx_base = np.argmax(histogram[:midpoint])
  # -> the max index of max point from midpoint to the end of histogram
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint

  # lets now store the left_base and right_base into new variables to update them without any errors
  leftx_current = leftx_base
  rightx_current = rightx_base

  # initially we are above the height of the image by 0.5 window height  = 370 "350 + 20" window_height = 40
  y_current = img.shape[0] + window_height//2

  # Create empty lists to reveice left and right lane pixel
  leftx, lefty, rightx, righty = [], [], [], []

  # Step through the windows one by one
  for _ in range(nwindows):
    # center of the current window (from top to bottom)
    y_current -= window_height
    # we are operating the left and right window in parallel,
    # we are starting from the center of the first window of left and right lanes starting from top to bottm
    center_left = (leftx_current, y_current)
    center_right = (rightx_current, y_current)

    good_left_x, good_left_y = pixels_in_window(center_left, margin, window_height)
    good_right_x, good_right_y = pixels_in_window(center_right, margin, window_height)

    # Append these indices to the lists
    leftx.extend(good_left_x)
    lefty.extend(good_left_y)
    rightx.extend(good_right_x)
    righty.extend(good_right_y)

    if len(good_left_x) > minpix:
        leftx_current = np.int32(np.mean(good_left_x))
    if len(good_right_x) > minpix:
        rightx_current = np.int32(np.mean(good_right_x))

  return leftx, lefty, rightx, righty, out_img

def fit_poly(img):
  """Find the lane line from an image and draw it.

  Parameters:
      img (np.array): a binary warped image

  Returns:
      out_img (np.array): a RGB image that have lane line drawn on that.
  """
  global left_fit
  global right_fit

  # we have all pixels in each window in left and right lanes
  leftx, lefty, rightx, righty, out_img = find_lane_pixels(img)

  if len(lefty) > 1500:
    left_fit = np.polyfit(lefty, leftx, 2) # 2nd degree polynomial 
    # print(left_fit)
  if len(righty) > 1500:
    right_fit = np.polyfit(righty, rightx, 2)

  # Generate x and y values for plotting
  maxy = img.shape[0] - 1
  miny = img.shape[0] // 3
  if len(lefty):
    maxy = max(maxy, np.max(lefty))
    miny = min(miny, np.min(lefty))

  if len(righty):
    maxy = max(maxy, np.max(righty))
    miny = min(miny, np.min(righty))

  ploty = np.linspace(miny, maxy, img.shape[0])

  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  # Visualization
  for i, y in enumerate(ploty):
    l = int(left_fitx[i])
    r = int(right_fitx[i])
    y = int(y)
    cv2.line(out_img, (l, y), (r, y), (0, 255, 0))

  lR, rR, pos = measure_curvature()
  return out_img

def measure_curvature():
  global left_fit
  global right_fit

  ym = 30/720
  xm = 3.7/700

  left_fit = left_fit.copy()
  right_fit = right_fit.copy()

  y_eval = 700 * ym
  # Compute R_curve (radius of curvature)
  left_curveR =  ((1 + (2*left_fit[0] *y_eval + left_fit[1])**2)**1.5)  / np.absolute(2*left_fit[0])
  right_curveR = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

  xl = np.dot(left_fit, [700**2, 700, 1])
  xr = np.dot(right_fit, [700**2, 700, 1])
  pos = (1280//2 - (xl+xr)//2)*xm
  return left_curveR, right_curveR, pos 

def plot(out_img):
  global dir
  global left_fit
  global right_fit
  global left_curve_img
  global right_curve_img
  global keep_straight_img
  global left_curve_img
  global right_curve_img
  global keep_straight_img

  np.set_printoptions(precision=6, suppress=True)
  
  lR, rR, pos = measure_curvature()

  value = None
  if abs(left_fit[0]) > abs(right_fit[0]):
    value = left_fit[0]
  else:
    value = right_fit[0]

  if abs(value) <= 0.00015:
    dir.append('F')
  elif value < 0:
    dir.append('L')
  else:
    dir.append('R')
  
  if len(dir) > 10:
    dir.pop(0)

  W = 400
  H = 300
  widget = np.copy(out_img[:H, :W])
  widget //= 2
  widget[0,:] = [0, 0, 255]
  widget[-1,:] = [0, 0, 255]
  widget[:,0] = [0, 0, 255]
  widget[:,-1] = [0, 0, 255]
  out_img[:H, :W] = widget

  direction = max(set(dir), key = dir.count)
  curvature_msg = "Curvature = {:.0f} m".format(min(lR, rR))
  if direction == 'F':
    straight_msg= "Curvature = 0 m"
    cv2.putText(out_img, straight_msg, org=(10, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

  if direction in 'LR':
    cv2.putText(out_img, curvature_msg, org=(10, 140), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

  cv2.putText(
      out_img,
      "Vehicle is {:.2f} m away from center".format(pos),
      org=(10, 180),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
      fontScale=0.66,
      color=(255, 255, 255),
      thickness=2)

  return out_img

def pipeline(input_frame):
  out_img = np.copy(input_frame)
  input_frame = undistort(input_frame)
  input_frame = get_eye_bird_view(input_frame)
  input_frame = thresholding(input_frame)
  input_frame = detect_lane_lines(input_frame)
  input_frame = get_front_view(input_frame)
  out_img = cv2.addWeighted(out_img, 1, input_frame, 0.6, 0)
  out_img = plot(out_img)
  return out_img

def pipeline_debug(input_frame):
  height, width = 1080, 1920
  FinalScreen = np.zeros((height, width, 3), dtype=np.uint8)
  out_img = np.copy(input_frame)
  
  undistoted_frame = undistort(input_frame)
  cv2.putText(undistoted_frame, 'Undistorted Frame',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[0:360,1280:1920] = cv2.resize(undistoted_frame, (640,360), interpolation=cv2.INTER_AREA)

  bird_view_frame = get_eye_bird_view(undistoted_frame)
  cv2.putText(bird_view_frame, 'Bird Eye View Frame',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[360:720,1280:1920] = cv2.resize(bird_view_frame, (640,360), interpolation=cv2.INTER_AREA)

  threshold_frame = thresholding(bird_view_frame)
  thresholded_frame_cpy = np.copy(threshold_frame)
  thresholded_frame_cpy = np.dstack((thresholded_frame_cpy, thresholded_frame_cpy, thresholded_frame_cpy))
  cv2.putText(thresholded_frame_cpy, 'Thresholded Frame',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[720:1080,1280:1920] = cv2.resize(thresholded_frame_cpy, (640,360), interpolation=cv2.INTER_AREA)

  lane_line_frame = detect_lane_lines(threshold_frame)
  cv2.putText(lane_line_frame, 'Detected Lane Lines Frame',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[720:1080,640:1280] = cv2.resize(lane_line_frame, (640,360), interpolation=cv2.INTER_AREA)

  front_view_frame = get_front_view(lane_line_frame)
  front_view_frame_untext = get_front_view(lane_line_frame)
  cv2.putText(front_view_frame, 'Front view Frame',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[720:1080,0:640] = cv2.resize(front_view_frame, (640,360), interpolation=cv2.INTER_AREA)


  out_img = cv2.addWeighted(out_img, 1, front_view_frame_untext, 0.6, 0)
  out_img = plot(out_img)
  cv2.putText(out_img, 'Final Video',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
  FinalScreen[0:720,0:1280] = cv2.resize(out_img, (1280,720), interpolation=cv2.INTER_AREA)
  
  return FinalScreen 

def process_video(input_path, output_path):
  clip = VideoFileClip(input_path)
  out_clip = clip.fl_image(pipeline)
  out_clip.write_videofile(output_path, audio=False)

def process_video_debug(input_path, output_path):
  clip = VideoFileClip(input_path)
  out_clip = clip.fl_image(pipeline_debug)
  out_clip.write_videofile(output_path, audio=False)

def main():
  
  mode = sys.argv[1]
  input_path = sys.argv[2]
  output_path = sys.argv[3]

  if mode == '0':
    process_video(input_path, output_path)

  if mode == '1':
    process_video_debug(input_path, output_path)


if __name__ == "__main__":
  main()