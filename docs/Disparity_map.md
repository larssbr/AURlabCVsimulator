
Load in the image Pair


```python
import cv2
import numpy as np

# The filnames
dirPathLeft = "tokt1_L_181.jpg"
dirPathRight = "tokt1_R_181.jpg"

# load the image
imgsLeft = cv2.imread(dirPathLeft)
imgsRight = cv2.imread(dirPathRight)


cv2.imshow("imageLeft" , imgsLeft)
cv2.imshow("imageRight" , imgsRight)
cv2.waitKey(0)
```




    -1



### Run the DisparityImage class process

1 - Load Camera parameters

2 - undistort image pair

3 - make image pair gray scale

4 - block matching

5 - remmove error margin (from challenging claibration parameters)

###  Load the camera parameters

The camera parameters come from a camera calibration done in matlab.
We need theese camera parameters in order to undistort/rectify the image pair before we can use "Block Matching"



```python
#def loadCameraParameters(self):
# left
fxL = 2222.72426
fyL = 2190.48031

k1L = 0.27724
k2L = 0.28163
k3L = -0.06867
k4L = 0.00358
k5L = 0.00000

cxL = 681.42537
cyL = -22.08306

skewL = 0

p1L = 0
p2L = 0
p3L = 0
p4L = 0

# right
fxR = 2226.10095
fyR = 2195.17250

k1R = 0.29407
k2R = 0.29892
k3R = 0 - 0.08315
k4R = -0.01218
k5R = 0.00000

cxR = 637.64260
cyR = -33.60849

skewR = 0

p1R = 0
p2R = 0
p3R = 0
p4R = 0

# x0 and y0 is zero
x0 = 0
y0 = 0

intrinsic_matrixL = np.matrix([[fxL, skewL, x0], [0, fyL, y0], [0, 0, 1]])
intrinsic_matrixR = np.matrix([[fxR, skewR, x0], [0, fyR, y0], [0, 0, 1]])

distCoeffL = np.matrix([k1L, k2L, p1L, p2L, k3L])
distCoeffR = np.matrix([k1R, k2R, p1R, p2R, k3R])

# Parameters

base_offset = 30.5 # b:= base offset, (the distance *between* your cameras)
# f:= focal length of camera,
fx = 2222
focal_length = (fx * 35) / 1360  # 1360 is the width of the image, 35 is width of old camera film in mm (10^-3 m)

```

### Block matching

There is several important parameters:
- SADWindowSize
- preFilterType
- preFilterSize
- preFilterCap
- minDisparity
- numberOfDisparities
- textureThreshold
- uniquenessRatio
- speckleRange
- speckleWindowSize

It is important to note 
disparity = cv2.cv.CreateMat(c, r, cv2.cv.CV_32F)
cv2.cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)

disparity_visual = cv2.cv.CreateMat(c, r, cv2.cv.CV_8U)
cv2.cv.Normalize(disparity, disparity_visual, 0, 255, cv2.cv.CV_MINMAX)


Normalization



```python
def getDisparity(imgLeft, imgRight, method="BM"):
    # 1 make the images grayscale
    gray_left = cv2.cvtColor(imgLeft, cv2.cv.CV_BGR2GRAY)
    gray_right = cv2.cvtColor(imgRight, cv2.cv.CV_BGR2GRAY)
    
    # 2 preform "BM" --> "Block Matching"
    c, r = gray_left.shape
    if method == "BM":

        sbm = cv2.cv.CreateStereoBMState()
        disparity = cv2.cv.CreateMat(c, r, cv2.cv.CV_32F)
        sbm.SADWindowSize = 9
        sbm.preFilterType = 1
        sbm.preFilterSize = 5
        sbm.preFilterCap = 61
        sbm.minDisparity = -39
        sbm.numberOfDisparities = 112
        sbm.textureThreshold = 507
        sbm.uniquenessRatio = 0
        sbm.speckleRange = 8
        sbm.speckleWindowSize = 0


        gray_left = cv2.cv.fromarray(gray_left)
        gray_right = cv2.cv.fromarray(gray_right)

        cv2.cv.FindStereoCorrespondenceBM(gray_left, gray_right, disparity, sbm)
        disparity_visual = cv2.cv.CreateMat(c, r, cv2.cv.CV_8U)
        cv2.cv.Normalize(disparity, disparity_visual, 0, 255, cv2.cv.CV_MINMAX)

        disparity_visual = np.array(disparity_visual)
    
    elif method == "SGBM":
        sbm = cv2.StereoSGBM()
        sbm.SADWindowSize = 9
        sbm.numberOfDisparities = 96
        sbm.preFilterCap = 63
        sbm.minDisparity = -21
        sbm.uniquenessRatio = 7
        sbm.speckleWindowSize = 0
        sbm.speckleRange = 8
        sbm.disp12MaxDiff = 1
        sbm.fullDP = False

        disparity = sbm.compute(gray_left, gray_right)
        disparity_visual = cv2.normalize(disparity, alpha=0, beta=255, norm_type=cv2.cv.CV_MINMAX,
                                         dtype=cv2.cv.CV_8U)
    
    return disparity_visual
```

Doing it all in a pipeline: 

 Undistort images
 
 


```python
def disparityCalc():
    # 1 undistort the image pair
    undistorted_image_L = cv2.undistort(imgsLeft, intrinsic_matrixL, distCoeffL, None)
    undistorted_image_R = cv2.undistort(imgsRight, intrinsic_matrixR, distCoeffR, None)
    
    # 2 --> calculate disparity images
    disparity_visual = getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")
    disparity_visual = disparity_visual.astype(np.uint8)
    return disparity_visual
```

Display the disparity image 


```python
disparity_visual = disparityCalc()
cv2.imshow("disparity", disparity_visual)
cv2.waitKey(0)
```




    -1



#### Remove the error margin


```python
disparity_visual = disparityCalc()

#disparity_visual = disparity_visual.astype(np.float32)

# this part is to remove the error from the calibration
width, height = disparity_visual.shape[:2][::-1]
margin = 200

y1 = 0
y2 = height
x1 = margin
x2 = width - margin
# "Croping the image"
disparity_visual = disparity_visual[y1:y2, x1:x2]

cv2.imshow("disparity", disparity_visual)
cv2.waitKey(0)
```




    -1




```python

```


```python

```
