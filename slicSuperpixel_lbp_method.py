import cv2
import numpy as np
# import the necessary packages
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import feature

from skimage.segmentation import felzenszwalb

# Classifier
from sklearn.svm import LinearSVC

# profiling the code
import cProfile

# to save and load, the model that is created from the classification
from sklearn.externals import joblib

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

class modelTools(object):

	def __init__(self, createdModel, imageOcean, imageOther):
		self.imageOcean =  self.resizeImage(imageOcean)
		self.imageOther =  self.resizeImage(imageOther)

		if createdModel == True:
			self.model = self.loadModel()

		elif createdModel == False:
			self.model = self.createModel()
			# need to load the model after it is created
			self.model = self.loadModel()

	def saveModel(self, model):
		joblib.dump(model, "model/filename_model.pkl")

	def loadModel(self):
		return joblib.load("model/filename_model.pkl")

	def createModel(self):
		dataClassOcean = analyseROITools(self.imageOcean, "ocean")
		data, labels = dataClassOcean.get_HistofContoursOfSegments("ocean")

		dataClassOther = analyseROITools(self.imageOther, "other")

		dataOther, labelsOther = dataClassOther.get_HistofContoursOfSegments("other")

		dataList = data + dataOther
		labelsList = labels + labelsOther

		# Train a Linear SVM on the data
		model = LinearSVC(C=100.0, random_state=42)
		model.fit(dataList, labelsList)

		self.saveModel(model)

	def get_model(self):
		return self.model

	def resizeImage(self, image):
		(h, w) = image.shape[:2]

		width = 360  #  This "width" is the width of the resize`ed image
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
		return resized

class analyseROITools(object):

	def __init__(self, image, labelName):

		self.image = image
		self.segments = slic(img_as_float(self.image), n_segments=100, sigma=5)
		#self.segments = felzenszwalb(img_as_float(self.image), scale=3.0, sigma=0.95, min_size=5)

		self.imageROIList = self.get_ROIofContoursList(image, self.segments)
		self.labelName = labelName

		#self.desc = LocalBinaryPatterns(24, 8)  # numPoints = 24, radius = 8
		#self.desc = LocalBinaryPatterns(10, 5)  # numPoints = 24, radius = 8
		self.desc = LocalBinaryPatterns(10, 5)

		#self.data, self.labels = self.get_HistofContoursOfSegments()

	def get_ROIofContoursList(self, image, segments):
		# 1. Loop over each superpixel segment and extract its contour.
		# 2. Compute bounding box of contour.
		# 3. Extract the rectangular ROI.

		imageROIList = []
		for (i, segVal) in enumerate(np.unique(segments)):
			# construct a mask for the segment
			print "[x] inspecting segment %d" % (i)
			mask = np.zeros(image.shape[:2], dtype="uint8")
			mask[segments == segVal] = 255

			# threshold the image, then perform a series of erosions +
			# dilations to remove any small regions of noise
			thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.erode(thresh, None, iterations=2)
			thresh = cv2.dilate(thresh, None, iterations=4)

			# calling the cv2.findContours on a treshold of the image
			(contours0, _) = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for ctr in contours0:
				# box = cv2.minAreaRect(ctr)
				# (x, y, w, h) = cv2.minAreaRect(ctr)
				# 2 compute bounding box of countour
				(x, y, w, h) = cv2.boundingRect(ctr)

				# 3 extract the rectangular ROI
				# extract the ROI from the image and draw a bounding box
				# surrounding the MRZ
				imageROI = image[y:y + h, x:x + w].copy()
				imageROIList.append(imageROI)
			# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		return imageROIList

	def get_HistofContoursOfSegments(self, labelName):
		# initialize the local binary patterns descriptor along with
		# the data and label lists
		desc = self.desc
		# initialize the image descriptor -- a 2D LAB histogram using A and B channel with 8 bins per channel
		#desc = LABHistogram([8, 8])
		data = []
		labels = []

		for imageROI in self.imageROIList:
			# 4 pass that into descriptor to obtain feature vector.
			grayImage = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
			hist = desc.describe(grayImage)

			# update the label and data lists
			data.append(hist)
			labels.append(labelName)

		return data, labels

class predictionTool(object):

	def __init__(self, image, model, radiusTresh, isObstacleInfront_based_on_radius):

		self.image = self.resizeImage(image)
		self.segments = slic(img_as_float(self.image), n_segments=100, sigma=5)
		#self.segments = felzenszwalb(img_as_float(self.image), scale=3.0, sigma=0.95, min_size=5)

		#self.image = image
		self.model = model
		self.radiusTresh = radiusTresh
		#self.h, self.w = image.shape[:2]

		self.centerCordinates = []
		#self.desc = LocalBinaryPatterns(24, 8)
		#self.desc = LocalBinaryPatterns(10, 5)
		self.desc = LocalBinaryPatterns(10, 5)

		#self.imageROIList = self.get_ROIofContoursList()
		#self.mask = np.zeros(self.image.shape[:2], dtype="uint8")
		self.imageROIList, self.centerList, self.predictionList, self.maskedImage = self.extractROIofSegmentandCenterList()

	# self.data, self.labels = self.get_HistofContoursOfSegments()

	def resizeImage(self, image):
		(h, w) = image.shape[:2]
		#width = 1360
		width = 360
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
		return resized

	def get_ROIofContoursList(self):
		# 1. Loop over each superpixel segment and extract its contour.
		# 2. Compute bounding box of contour.
		# 3. Extract the rectangular ROI.

		imageROIList = []
		#mask = np.zeros(self.image.shape[:2], dtype="uint8")
		for (i, segVal) in enumerate(np.unique(self.segments)):
			# construct a mask for the segment
			#print "[x] inspecting segment %d" % (i)
			mask = np.zeros(self.image.shape[:2], dtype="uint8")
			mask[self.segments == segVal] = 255

			# threshold the image, then perform a series of erosions +
			# dilations to remove any small regions of noise
			thresh = cv2.threshold(mask, 45, 255, cv2.THRESH_BINARY)[1]
			thresh = cv2.erode(thresh, None, iterations=2)
			thresh = cv2.dilate(thresh, None, iterations=4)

		# calling the cv2.findContours on a treshold of the image
		contours0, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		# moments = [cv2.moments(cnt) for cnt in contours0]

		# rounded the centroids to integer.
		# centroids = [(int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))) for m in moments]

		# print 'len(contours0)'
		# print len(contours0)
		for ctr in contours0:
			# 2 compute bounding box of countour
			(x, y, w, h) = cv2.boundingRect(ctr)

			# 3 extract the rectangular ROI
			imageROI = self.image[y:y + h, x:x + w].copy()
			imageROIList.append(imageROI)

		return imageROIList

	def extractROIofSegmentandCenterList(self):
		#desc = LocalBinaryPatterns(24, 8)
		centerList = []
		imageROIList =[]
		predictionList = []
		# create mask
		# This mask has the same width and height a the original image and has a default value of 0 (black).
		maskedImage = np.zeros(self.image.shape[:2], dtype="uint8")

		test = 1
		test = self.segments
		test = 2

		# loop over the unique segment values
		for (i, segVal) in enumerate(np.unique(self.segments)):
			# construct a mask for the segment
			print "[x] inspecting segment %d" % (i)
			mask = np.zeros(self.image.shape[:2], dtype="uint8")
			mask[self.segments == segVal] = 255
			imageMasked = cv2.bitwise_and(self.image, self.image, mask=mask)
			grayImage_masked = cv2.cvtColor(imageMasked, cv2.COLOR_BGR2GRAY)

			# calling the cv2.findContours on a treshold of the image
			contours0, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			#moments = [cv2.moments(cnt) for cnt in contours0]

			# rounded the centroids to integer.
			#centroids = [(int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))) for m in moments]

			for ctr in contours0:
				# 1 compute bounding box of countour
				(x, y, w, h) = cv2.boundingRect(ctr)

				# 2 add the center position of this bounding box in the centerList
				ctr_center_pos = ((x+w/2) , (y+h/2))
				centerList.append(ctr_center_pos)

				# 3 extract the rectangular ROI
				imageROI = self.image[y:y + h, x:x + w].copy()

				# Mask the imageROI here according to prediction
				# 4 pass that into descriptor to obtain feature vector.

				grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
				hist = self.desc.describe(grayImageROI)

				model = self.model
				if model == None:
					print "it was none"

				prediction = model.predict(hist)[0]
				predictionList.append(prediction)

				# construct a mask for the segment
				if prediction == "other":
					maskedImage[y:y + h, x:x + w] = 255

				if prediction == "ocean":
					maskedImage[y:y + h, x:x + w] = 0

				imageROIList.append(grayImageROI)

		return imageROIList, centerList, predictionList, maskedImage

	def get_maskedImage(self):
		return self.maskedImage

	def show_maskedImage(self):
		cv2.imshow('mask', self.maskedImage)
		cv2.waitKey(1)

	# unused method
		def findCentroid(self, imgBW):

			imgBWCopy = imgBW.astype(np.uint8)

			(contours0, _) = cv2.findContours(imgBWCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			centerCordinates = []
			try:
				moments = [cv2.moments(cnt) for cnt in contours0]
				# rounded the centroids to integer.
				centroids = [(int(round(m['m10'] / m['m00'])), int(round(m['m01'] / m['m00']))) for m in moments]
				for ctr in centroids:
					# draw a black little empty circle in the centroid position
					centerCircle_Color = (0, 0, 0)
					# cv2.circle(imgBW, tuple(ctr), 4, centerCircle_Color)
					centerCordinates.append(ctr)
			except:
				for ctr in contours0:
					# draw a black little empty circle in the centroid position
					centerCircle_Color = (0, 0, 0)
					# cv2.imshow("imgBw", imgBW)
					# cv2.waitKey(0)
					# cv2.circle(imgBW, tuple(ctr), 4, centerCircle_Color)
					centerCordinates.append(ctr)

			return imgBW, centerCordinates

	######################################################################################
	# used for visualization of the prediction in the main() method
	def showPredictionOutput(self):
		image_withText = self.image.copy()

		# show the output of the prediction with text
		for (i, segVal) in enumerate(np.unique(self.segments)):
			CORD = self.centerList[i]
			if self.predictionList[i] == "other":
				colorFont = (255, 0, 0) # "Blue color for other"
			else:
				colorFont = (0, 0, 255) # "Red color for ocean"

			#textOrg = CORD
			#textOrg = tuple(numpy.subtract((10, 10), (4, 4)))

			testOrg = (40,40) # need this for the if statment bellow

			# for some yet unknown reason CORD does sometime contain somthing like this [[[210 209]] [[205 213]] ...]
			# the following if statemnet is to not get a error becouse of this
			if len(CORD) == len(testOrg):
				#textOrg = tuple(np.subtract(CORD, (12, 0)))
				textOrg = CORD
				cv2.putText(self.image, self.predictionList[i], textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.1, colorFont, 3)
				markedImage = mark_boundaries(img_as_float(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)), self.segments)
			else:
				pass

		cv2.imshow("segmented image", markedImage)
		cv2.waitKey(0)


def main():
	'''
	This function is here for testing purpuses for just the slicSuperpixel_lbp_method.py script.
	So you can run the algorithm on one image, instead of a "stream" of images
	'''

	#1 ######  Set the parameters
	createdModel = False
	isObstacleInfront_based_on_radius = False
	radiusTresh = 30

	#2   ###### Get or create model ##################################
	imageOcean = cv2.imread("tokt1_R_1037.jpg")
	imageOther = cv2.imread("raptors.png")
	modelClass = modelTools(createdModel, imageOcean, imageOther)
	model = modelClass.get_model()

	#3    #### Choce image to predict #####################################
	# image = cv2.imread("tokt1_R_267.jpg")
	#image = cv2.imread("transpondertowerIMG/tokt1_L_473.jpg")
	image = cv2.imread(r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\repeatExperiment\Left\tokt1_L_179.jpg")
	#image = cv2.imread(r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\repeatExperiment\Left\tokt1_L_154.jpg")
	#image = cv2.imread("transpondertowerIMG/tokt1_L_473.jpg")

	#4   ####  use model to predict a new image #### Predict image ########## test the prediction of the model ################
	predictionClass = predictionTool(image, model, radiusTresh, isObstacleInfront_based_on_radius)
	image = predictionClass.get_maskedImage()

	#5   ### Display image ##############################
	cv2.imshow("image", image)
	cv2.waitKey(0)

	# 6 ### Display prediction with text ##################
	predictionClass.showPredictionOutput(


	)

if __name__ == '__main__':
    cProfile.run('main()')

