# this method is way fater than the superpixel_haralick_method
# instead of using segmentation it extracts 100 smaller image ROI of an resized image.
# it then uses this for training the model. And to predict the model.

import cv2
import numpy as np
# import the necessary packages
#from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
#from skimage import feature

# for the Haralick descriptor
import mahotas

from skimage.segmentation import felzenszwalb

# Classifier
from sklearn.svm import LinearSVC

# profiling the code
import cProfile

# to save and load, the model that is created from the classification
from sklearn.externals import joblib

# Feature extraction
class Haralick:
	# this mehtod is a Harlick discriptor
	# it uses the mahotas library

	def describe(self, image, eps=1e-7):
		# it should send in a grayscale image in the describe function
		#extract Haralick texture features in 4 directions, then take the
		# mean of each direction
		# ignore_zeros=True since i have masked the image, and therefore we want to ignore black color == 0 (zeroes)
		features = mahotas.features.haralick(image, ignore_zeros=True).mean(axis=0)

		# return the haralick feature
		return features

class modelTools(object):
	def __init__(self, createdModel, imageOcean, imageOther):
		self.imageOcean =  self.resizeImage(imageOcean)
		self.imageOther =  self.resizeImage(imageOther)

		self.desc = Haralick()

		if createdModel == True:
			self.model = self.loadModel()

		elif createdModel == False:
			self.model = self.createModel()
			# need to load the model after it is created
			self.model = self.loadModel()

	def resizeImage(self, image):
		(h, w) = image.shape[:2]
		width = 360  # This "width" is the width of the resize`ed image
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (width, int(h * r))
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
		return resized

	def extractHistogramList(self, image, label):

		histogramList = []
		LabelList = []

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

			x = 0  # it starts at 0 for a new row
			for j in xrange(10):
				print "[x] inspecting imageROI %d" % (counterInt)
				counterInt = counterInt + 1

				x = cellSizeXdir * (j)

				imageROI = image[y: cellSizeYdir * (i + 1), x:cellSizeXdir * (j + 1)]

				# print "ystart  " + str(y) + "  yjump  " + str((cellSizeYdir * (i + 1)))
				# print "xstart  " + str(x) + "  xjump  " + str((cellSizeXdir * (j + 1)))

				# grayscale and calculate histogram
				grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
				hist = self.desc.describe(grayImageROI)

				histogramList.append(hist)
				LabelList.append(label)

		return histogramList, LabelList


	def createModel(self):
		#dataClassOcean = analyseROITools(self.imageOcean, "ocean")
		#data, labels = dataClassOcean.extractHistogramList("ocean")

		data, labels = self.extractHistogramList(self.imageOcean,"ocean")

		#dataClassOther = analyseROITools(self.imageOther, "other")
		#dataOther, labelsOther = dataClassOther.extractHistogramList("other")

		dataOther, labelsOther = self.extractHistogramList(self.imageOther,"other")

		dataList = data + dataOther
		labelsList = labels + labelsOther

		# Train a Linear SVM on the data
		model = LinearSVC(C=100.0, random_state=42)
		model.fit(dataList, labelsList)

		self.saveModel(model)

	# load and save
	def saveModel(self, model):
		joblib.dump(model, "model/filename_model.pkl")

	def loadModel(self):
		return joblib.load("model/filename_model.pkl")

	def get_model(self):
		# return model
		return self.model

class predictionTool(object):

	def __init__(self, image, model, radiusTresh, isObstacleInfront_based_on_radius):

		self.image = self.resizeImage(image)

		self.model = model
		self.radiusTresh = radiusTresh
		#self.h, self.w = image.shape[:2]

		self.centerCordinates = []
		#self.desc = LocalBinaryPatterns(24, 8)
		#self.desc = LocalBinaryPatterns(10, 5)
		self.desc = Haralick()

		self.maskedImage, self.predictionList = self.predictMaskedImage()


	def predictMaskedImage(self):
		imageROIList = []
		predictionList = []

		# This mask has the same width and height a the original image and has a default value of 0 (black).
		maskedImage = np.zeros(self.image.shape[:2], dtype="uint8")
		########### create imageROIList here ############

		(h, w) = self.image.shape[:2]

		# Divide the image into 100 pieces
		cellSizeYdir = h / 10
		cellSizeXdir = w / 10

		# start in origo
		x = 0
		y = 0
		counterInt = 0

		#######################################

		# 10*10 = 100
		for i in xrange(10):

			# update this value
			y = cellSizeYdir * (i)

			x = 0  # it starts at 0 for a new row
			for j in xrange(10):
				print "[x] inspecting imageROI %d" % (counterInt)
				counterInt = counterInt + 1

				x = cellSizeXdir * (j)

				imageROI = self.image[y: cellSizeYdir * (i + 1), x:cellSizeXdir * (j + 1)]

				#print "ystart  " + str(y) + "  yjump  " + str((cellSizeYdir * (i + 1)))
				#print "xstart  " + str(x) + "  xjump  " + str((cellSizeXdir * (j + 1)))

				#########################################

				# grayscale and calculate histogram
				grayImageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY)
				hist = self.desc.describe(grayImageROI)

				# need prediction to mask image
				model = self.model
				if model == None:
					print "it was none"

				# reshape the histogram to work with sci kit learn
				histNew = np.reshape(hist, (1, len(hist)))

				prediction = model.predict(histNew)[0]
				predictionList.append(prediction)
				# prediction = model.predict(hist)[0]


				# HERE the returned maskedImage is created
				# construct a mask for the segment
				if prediction == "other":
					maskedImage[y:y + cellSizeYdir, x:x + cellSizeXdir] = 255

				if prediction == "ocean":
					maskedImage[y:y + cellSizeYdir, x:x + cellSizeXdir] = 0

		return maskedImage, predictionList

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


	def get_maskedImage(self):
		return self.maskedImage

	def show_maskedImage(self):
		cv2.imshow('mask', self.maskedImage)
		cv2.waitKey(1)

	# unused method
		def findCentroid(self, imgBW):

			imgBWCopy = imgBW.astype(np.uint8)

			# contours0, hierarchy = cv2.findContours( imgBW.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			contours0, hierarchy = cv2.findContours(imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

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
	def showPredictionOutputonROI(self):
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
			# the following if statment is to not get a error becouse of this
			if len(CORD) == len(testOrg):
				#textOrg = tuple(np.subtract(CORD, (12, 0)))
				textOrg = CORD
				cv2.putText(self.image, self.predictionList[i], textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.1, colorFont, 3)
				markedImage = mark_boundaries(img_as_float(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)), self.segments)
			else:
				pass

		cv2.imshow("segmented image", markedImage)
		cv2.waitKey(0)

		def showPredictionOutputonROI(self):
			centerList = []

			image_withText = self.image.copy()

			(h, w) = self.image.shape[:2]

			# Divide the image into 100 pieces
			cellSizeYdir = h / 10
			cellSizeXdir = w / 10

			for i in xrange(10):

				# update this value
				y = cellSizeYdir * (i)

				x = 0  # it starts at 0 for a new row
				for j in xrange(10):
					print "[x] inspecting imageROI %d" % (counterInt)
					counterInt = counterInt + 1

					x = cellSizeXdir * (j)

					imageROI = self.image[y: cellSizeYdir * (i + 1), x:cellSizeXdir * (j + 1)]
					center = (y + (cellSizeYdir * (i + 1))/2, x + (cellSizeXdir * (j + 1))/2)
					centerList.append(center)


					# Draw boxes around things

					# loop over the y-axis of the image
					# draw a line from the current y-coordinate to the right of
					# the image
					cv2.line(stacked, (0, y), (w, cellSizeYdir * (i + 1)), (0, 255, 0), 1)

					# draw a line from the current x-coordinate to the bottom of
					# the imagez
					cv2.line(stacked, (x, 0), (x, cellSizeXdir * (j + 1)), (0, 255, 0), 1)

					# create the 3D grayscale image --> so that I can make color squares for figures to the thesis
					# This does not change the histograms created.
					stacked = np.dstack([gray] * 3)

					# Draw the box around area
					# loop over the x-axis of the image
					for x in xrange(0, w, cellSize):
						# draw a line from the current x-coordinate to the bottom of
						# the imagez
						cv2.line(stacked, (x, 0), (x, cellSizeXdir * (j + 1)), (0, 255, 0), 1)

					# loop over the y-axis of the image
					for y in xrange(0, h, cellSize):
						# draw a line from the current y-coordinate to the right of
						# the image
						cv2.line(stacked, (0, y), (w, y), (0, 255, 0), 1)

					# draw a line at the bottom and far-right of the image
					cv2.line(stacked, (0, h - 1), (w, h - 1), (0, 255, 0), 1)
					cv2.line(stacked, (w - 1, 0), (w - 1, h - 1), (0, 255, 0), 1)



			# show the output of the prediction with text
			for (i, predVal) in enumerate(self.predictionList):
				CORD = centerList[i]
				if self.predictionList[i] == "other":
					colorFont = (255, 0, 0)  # "Blue color for other"
				else:
					colorFont = (0, 0, 255)  # "Red color for ocean"

				# textOrg = CORD
				# textOrg = tuple(numpy.subtract((10, 10), (4, 4)))

				testOrg = (40, 40)  # need this for the if statment bellow

				# for some yet unknown reason CORD does sometime contain somthing like this [[[210 209]] [[205 213]] ...]
				# the following if statment is to not get a error becouse of this
				if len(CORD) == len(testOrg):
					# textOrg = tuple(np.subtract(CORD, (12, 0)))
					textOrg = CORD
					cv2.putText(self.image, self.predictionList[i], textOrg, cv2.FONT_HERSHEY_SIMPLEX, 0.1, colorFont,
								3)
					markedImage = mark_boundaries(img_as_float(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)),
												  self.segments)
					# Draw boxes around things


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
	predictionClass.showPredictionOutput()

if __name__ == '__main__':
    cProfile.run('main()')

