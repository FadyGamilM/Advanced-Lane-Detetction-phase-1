# IMPORT ESSENTIAL LIBRARIES
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from docopt import docopt
from moviepy.editor import VideoFileClip
import glob

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
  frames = glob.glob('/content/drive/MyDrive/camera_cal/calibration*.jpg')

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