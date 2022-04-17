# IMPORT ESSENTIAL LIBRARIES
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from docopt import docopt
from moviepy.editor import VideoFileClip
import glob
from google.colab.patches import cv2_imshow

# CALIBRATE THE CAMERA AND REMOVE DISTORTION
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