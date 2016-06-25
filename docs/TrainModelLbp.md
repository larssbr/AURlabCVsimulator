
### Train model 

0 - import the necessary packages

1.1 - Load image of "ocean" and "other"

1.2 - resize image

2 - make 100 imageROI described as histogram out of each training images

3 - label the ones from the first image as "ocean" and the other ones for "other"

4 - calculate the feature descriptor

5 - send the quatified feature vector together with the label "name" in into the SVMC

6 - save the created model

0 -- import the necessary packages


```python
# import the necessary packages

# for the lbp
from skimage import feature

# Classifier
from sklearn.svm import LinearSVC

# to save and load, the model that is created from the classification
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2
```

1 - Load image of "ocean" and "other"


```python
def resizeImage(image):
    (h, w) = image.shape[:2]

    width = 360  #  This "width" is the width of the resize`ed image
    # calculate the ratio of the width and construct the
    # dimensions
    ratio = width / float(w)
    dim = (width, int(h * ratio))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    #resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)
    return resized
```


```python

# 1 load the image
imagepathOcean = r"trainingIMG/ocean.jpg"
imageOcean = cv2.imread(imagepathOcean)


imagepathOther = r"trainingIMG/NemoReef.jpg"
imageOther = cv2.imread(imagepathOther)

# 2 resize the image
imageOcean = resizeImage(imageOcean)
(h, w) = imageOcean.shape[:2]
cellSize = h/10

imageOther = resizeImage(imageOther)
(h, w) = imageOther.shape[:2]
cellSize = h/10

# 3 convert the image to grayscale and show it
imageOceanGray = cv2.cvtColor(imageOcean, cv2.COLOR_BGR2GRAY)
cv2.imshow("ImageoceanGray", imageOceanGray)
cv2.waitKey(0)

imageOtherGray = cv2.cvtColor(imageOther, cv2.COLOR_BGR2GRAY)
cv2.imshow("ImageotherGray", imageOtherGray)
cv2.waitKey(0)

# save the image
cv2.imwrite("docsIMG/gray_resized_imageOcean.png", imageOceanGray)
cv2.imwrite("docsIMG/gray_resized_imageOther.png", imageOtherGray)

```




    True



The image displayed
![gray_resized_imageOcean](docsIMG/gray_resized_imageOcean.png)

![gray_resized_imageOther](docsIMG/gray_resized_imageOther.png)

2 - make 100 imageROI described as histogram out of each training images

3 - label the ones from the first image as "ocean" and the other ones for "other"

4 - calculate the feature descriptor



```python
# Feature extraction
class LocalBinaryPatterns(object):
    # this class is from: http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # Compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method= "uniform" ) # method= "ror") #method="var")# method="nri_uniform")  # method="uniform")
        # using unifrom binary pattern (watch this to understand better): https://www.youtube.com/watch?annotation_id=annotation_98709127&feature=iv&src_vid=wpAwdsubl1w&v=v-gkPTvdgYo
        # different merhod= --> http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=local_binary_pattern#skimage.feature.local_binary_pattern
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

        # Normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist
```


```python
def extractHistogramList(image, label):

    # get the descriptor class initiated
    desc = LocalBinaryPatterns(10, 5)
        
    histogramList = []
    LabelList = []

    # This mask has the same width and height a the original image and has a default value of 0 (black).
    maskedImage = np.zeros(image.shape[:2], dtype="uint8")
    ########### create imageROIList here ############

    (h, w) = image.shape[:2]

    # Divide the image into 100 pieces
    cellSizeYdir = h / 10
    cellSizeXdir = w / 10

    # start in origo
    x = 0
    y = 0
    counterInt = 0

    # 10*10 = 100
    for i in xrange(10):

        # update this value
        y = cellSizeYdir * (i)
        
        x = 0 # it starts at 0 for a new row
        for j in xrange(10):
            #print "[x] inspecting imageROI %d" % (counterInt)
            counterInt = counterInt + 1
            
            x = cellSizeXdir * (j)
            
            imageROI = image[y: cellSizeYdir * (i+1), x:cellSizeXdir * (j+1)]
            
            #print "ystart  " + str(y) + "  yjump  " + str((cellSizeYdir * (i+1)))
            #print "xstart  " + str(x) +  "  xjump  " + str((cellSizeXdir * (j+1)))

            # grayscale and calculate histogram
            grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
            hist = desc.describe(grayImageROI)
            
            histogramList.append(hist)
            LabelList.append(label)
            
    return histogramList, LabelList                                               
```


```python

# describe ocean
histogramListOcean, oceanLabelList = extractHistogramList(imageOcean, "ocean")

# describe other
histogramListOther, otherLabelList = extractHistogramList(imageOther, "other")

```

5 - send the quantified feature vector together with the label "name" in into the SVMC


```python
dataList = histogramListOcean + histogramListOther
labelsList = oceanLabelList + otherLabelList

# Train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(dataList, labelsList)

#self.saveModel(model)
joblib.dump(model, "model/filename_model.pkl")
```




    ['model/filename_model.pkl',
     'model/filename_model.pkl_01.npy',
     'model/filename_model.pkl_02.npy',
     'model/filename_model.pkl_03.npy']




```python

```
