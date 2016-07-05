from pymba import *
import numpy as np
import cv2
import time

import math # to get math.pi value
import socket # to send UDP message to labview pc

from collections import deque # to keep track of a que of last poitions of the center of an object
import datetime # to print the time to the textfile

from importDataset import Images  # Import the functions made in importDataset

from imageTools import centroidTools

# import the classes to do the superpixelMethod
from slicSuperpixel_lbp_method import modelTools
from slicSuperpixel_lbp_method import predictionTool

# for profiling the code
import cProfile

from simple_lbp_model import simpleModelTools
from simple_lbp_model import simpleAnalyseImageTools
from simple_lbp_model import simplePredictionTool

# HELPER FUNCTIONS degrees and radians
def rad2deg(radians):
    # degrees = 180 * radians / pi
    pi = math.pi
    degrees = 180 * radians / pi
    return degrees

def deg2rad(degrees):
    # radians = pi * degrees / 180
    pi = math.pi
    radians = pi * degrees / 180
    return radians

#################################################
class RecordImages(object):
    def __init__(self, imgLeft, imgRight, img_DISPARITY, img_depthMap, pairNumber, folderName_saveImages, toktName):
        self.imgLeft = imgLeft
        self.imgRight = imgRight
        self.imgNameString_L = folderName_saveImages + "/" + toktName + "_L_" + str(pairNumber) + ".jpg"
        self.imgNameString_R = folderName_saveImages + "/" + toktName + "_R_" + str(pairNumber) + ".jpg"
        self.imgNameString_DISTANCE = folderName_saveImages + "/" + toktName + "_Depth_map_" + str(pairNumber) + ".jpg"
        self.imgNameString_DISPARITY = folderName_saveImages + "/" + toktName + "_Disp_map_" + str(pairNumber) + ".jpg"

    def saveImages(self):
        ############## save the images that has been used to create disparity######################
        cv2.imwrite(self.imgNameString_L, self.imgLeft)
        cv2.imwrite(self.imgNameString_R, self.imgRight)

        ############## save the disparity that has been created ####################################
        cv2.imwrite(self.imgNameString_DISTANCE, self.img_depthMap)
        cv2.imwrite(self.imgNameString_DISPARITY, self.img_DISPARITY)

class TalkUDP(object):
    # Comunication methods

    def __init__(self, MESSAGE) :#, disparity_visual):
        #self.MESSAGE = ObstacleAvoidance.getMessage()
        self.MESSAGE = MESSAGE

    def sendUDPmessage(self):

        # addressing information of target
        # TODO: extreact this to GUI level of code
        UDP_IP = "127.0.0.1"    # correct IP if only ONE computer is used --> it becomes very slow if it does not find an ip and this is the computers ip
        #UDP_IP = "192.168.1.73" # CORRECT IP TO THE AUR LAB DELL COMPUTER
        UDP_PORT = 1130

        print "message:", self.MESSAGE

        # initialize a socket, think of it as a cable
        # SOCK_DGRAM specifies that this is UDP
        try:
            sock = socket.socket(socket.AF_INET,  # Internet
                                 socket.SOCK_DGRAM)  # UDP
            # send the command
            sock.sendto(self.MESSAGE, (UDP_IP, UDP_PORT))

            # close the socket
            sock.close()
        except:
            pass

    def statusObjectInfront(self):
        ################################# UDP #########################################
        # Compare the 3 parts, (Left, Center, Right) with each other to find in what area the object is.
        # returnValue = compare3windows(depthMap, somthing )
        # Image ROI
        ####
        directionMessage = "status : "
        #####
        if self.isObsticleInFront(self.disparity_visual, self.isObsticleInFrontTreshValue):
            # if the treshold says there is somthing infront then change directions

            # directionMessage = "CALC"
            directionMessage = directionMessage + str(0) + " "
        else:  # if nothing is in front of camera, do not interupt the path
            # directionMessage = directionMessage + "CONTINUE"
            directionMessage = directionMessage + str(1) + " "

        print "directionMessage"
        return directionMessage

class DisparityImage(object):
    # This class calculates the disparity map from a left and right image
    # --- > Run the DisparityImage class process
    #1 - Load Camera parameters
    #2 - make image pair gray scale
    #3 - undistort image pair / rectifies images
    #4 - stereo block matching to calculate disparity
    #5 - remmove error margin (from challenging claibration parameters)

    def __init__(self, imgLeft, imgRight):
        self.intrinsic_matrixL = []
        self.intrinsic_matrixR = []
        self.distCoeffL = []
        self.distCoeffR = []
        self.focal_length = None
        self.base_offset = None

        self.imgLeft = imgLeft
        self.imgRight = imgRight
        self.start_time = time.time()
        self.dispTime = 0

        self.radiusTresh = 40
        self.folderName_saveImages = "savedImages"
        self.toktName = "tokt1"
        #self.object_real_world_mm = 500  # 1000mm = 1 meter to calculate distance to a known object.
        #self.isObsticleInFrontTreshValue = 1.7

        # self.loadCameraParameters()

    def loadCameraParameters(self):
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

        self.intrinsic_matrixL = np.matrix([[fxL, skewL, x0], [0, fyL, y0], [0, 0, 1]])
        self.intrinsic_matrixR = np.matrix([[fxR, skewR, x0], [0, fyR, y0], [0, 0, 1]])

        self.distCoeffL = np.matrix([k1L, k2L, p1L, p2L, k3L])
        self.distCoeffR = np.matrix([k1R, k2R, p1R, p2R, k3R])

        # Parameters

        self.base_offset = 30.5 # b:= base offset, (the distance *between* your cameras)
        # f:= focal length of camera,
        fx = 2222
        self.focal_length = (fx * 35) / 1360  # 1360 is the width of the image, 35 is width of old camera film in mm (10^-3 m)
        ############

        #return intrinsic_matrixL, intrinsic_matrixR, distCoeffL, distCoeffR, focal_length, base_offset

    def disparityCalc(self):
        ############# CALCULATE Disparity ############################
        # 1 make the images grayscale
        gray_left = cv2.cvtColor(self.imgLeft, cv2.cv.CV_BGR2GRAY)
        gray_right = cv2.cvtColor(self.imgRight, cv2.cv.CV_BGR2GRAY)

        # 2 undistort the image pair
        undistorted_image_L = cv2.undistort(gray_left, self.intrinsic_matrixL, self.distCoeffL, None)
        undistorted_image_R = cv2.undistort(gray_right, self.intrinsic_matrixR, self.distCoeffR, None)

        # 3 --> calculate disparity images
        disparity_visual = self.getDisparity(imgLeft=undistorted_image_L, imgRight=undistorted_image_R, method="BM")
        disparity_visual = disparity_visual.astype(np.uint8)
        return disparity_visual

    def getDisparity(self, imgLeft, imgRight, method="BM"):

        gray_left = cv2.cv.fromarray(imgLeft)
        gray_right = cv2.cv.fromarray(imgRight)

        print imgLeft.shape
        c, r = imgLeft.shape
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
            # disp = cv2.normalize(sgbm.compute(ri_l, ri_r), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        return disparity_visual

    # if camera is tilted, one can use this method
        # Imporve disparity image, by using a scale sin(20) to sin(50) --> becouse the camera is tilted 35 or 45 degrees?
        # make an array of values from sin(20) to sin(50)
        # disparity_visual_adjusted = self.camereaAngleAdjuster(disparity_visual)

    def process(self):
        # 1 load calibration parameters
        self.loadCameraParameters()

        start_time = time.time()
        dispTime = 0

        pairNumber = 0

        elapsed_time = time.time() - start_time + 1
        if (elapsed_time > dispTime):
            # print "creating disparity"
            disparity_visual = self.disparityCalc()

            disparity_visual = disparity_visual.astype(np.float32)

            # remove the error from the calibration
            width, height = disparity_visual.shape[:2][::-1]
            margin = 200
            # img= img[margin:width-margin , 0 : height]
            y1 = 0
            y2 = height
            x1 = margin
            x2 = width - margin
            disparity_visual = disparity_visual[y1:y2, x1:x2]

            return disparity_visual

class ObstacleAvoidance(object):

    def __init__(self, disparity_visual, isObsticleInFrontTreshValue, objectAVGCenter):
        self.disparity_visual = disparity_visual
        self.isObsticleInFrontTreshValue = isObsticleInFrontTreshValue
        self.objectAVGCenter = objectAVGCenter
        self.cx, self.cy = self.objectAVGCenter
        #self.center = center

        # get the dimensions of the image
        self.width, self.height = disparity_visual.shape[:2][::-1]
        self.middleX, self.middleY = (self.width/2), (self.height/2)

        #self.MESSAGE = None
        self.directionMessage = None

        # Send position of dangerous objects. To avoid theese positions.
        # this is for the average position of alle the picels the disparity captures.
        self.Xpos = self.findXposMessage()
        self.Ypos = self.findYposMessage()

        self.CORD = self.calcCORD()

        self.meanValue = self.meanPixelSUM(disparity_visual)
        self.MESSAGE = self.createMESSAGE()

    def get_centerPosMessage(self):
        # Send position of dangerous objects. To avoid theese positions.
        # this is for the average position of alle the pixels the disparity captures.
        Xpos = self.findXposMessage()
        Ypos = self.findYposMessage()
        print "Xpos"
        print Xpos
        # XposMessage = directionMessage + ' Xpos :'+ str(Xpos) +' Ypos :' + str(Ypos)
        #############
        centerPosMessage = 'Xpos : ' + str(Xpos) + '  Ypos : ' + str(Ypos)
        return centerPosMessage

    def findXposMessage(self):
        # make new "coordinate system"
        if self.cx < self.middleX:
            Xpos = - (self.middleX - self.cx)  # - 50 is a little to the left
        else:
            Xpos = self.cx - self.middleX

        return -Xpos  # - to send the direction the rov should go, more intutive then where it should not go

    def findYposMessage(self):
        # make new "coordinate system"
        if self.cy < self.middleY:
            Ypos = - (self.middleY - self.cx)  # - 50 is a little to the left
        else:
            Ypos = self.cx - self.middleY

        return -Ypos  # - to send the direction the rov should go, more intutive then where it should not go

    def meanPixelSUM(self, imgROI):
        '''
        sum = np.sum(imgROI)  #cv2.sumElems(imgROI)
        width, height = imgROI.shape[:2][::-1]
        totalNumberOfPixels = width * height
        mean = sum/totalNumberOfPixels
        '''
        meanValue = cv2.mean(imgROI)

        return meanValue[0]  # want to have the 0 array, else the data is represented liek this: (1.3585184021230532, 0.0, 0.0, 0.0)

    def isObsticleInFront(self):
        # This method takes the possible "object Image"
        # And calculates a meanPixelSUM and if this value is over a given
        # isObsticleInFrontTreshValue then it returns True, else False

        # meanValue = self.meanPixelSUM(img)
        print "meanValue for disparity image"
        print self.meanValue
        # if the meanValue is above a treshold for to "small areas of pixels in the image"
        if self.meanValue > self.isObsticleInFrontTreshValue:
            return True
        else:
            return False

    def createMESSAGE(self):
        directionMessage = "status : , "  # todo : uncoment it later
        # directionMessage = "status : "
        #####
        # --> tell path program
        # 1 if there is obstacle in the image
        # 0 if there is NO obstacle in the image
        # if the treshold says there is something infront then change directions
        status = int(self.isObsticleInFront())
        directionMessage = directionMessage + str(status) + " "

        print "directionMessage"
        print directionMessage

        print "Xpos"
        print self.Xpos
        # XposMessage = directionMessage + ' Xpos :'+ str(Xpos) +' Ypos :' + str(Ypos)
        #############
        # centerPosMessage = 'Xpos : ' + str(self.Xpos) + '  Ypos : ' + str(self.Ypos)
        # centerPosMessage = ' ,Xpos : ,' + str(self.Xpos) + ',  Ypos : , ' + str(self.Ypos)   # todo : uncoment it later

        centerPosMessage = 'Xpos : ' + str(self.Xpos) + '  Ypos : ' + str(self.Ypos)

        ##### add meanValue here ########
        meanValueMessage = ' , meanValue : ,' + str(self.meanValue)

        MESSAGE = directionMessage + centerPosMessage  + meanValueMessage
        return MESSAGE

    def get_MESSAGE(self):
        return self.MESSAGE

    def get_CORD(self):
        return self.CORD

    def calcCORD(self):
        # this method sets a cordinate for direction of the ROV
        # if Xpos is positive, then we want to move alot to the Right in the image
        # if Xpos is negative, then we want to move alot to the Left in the image
        if self.Xpos > 0:
            print "turn right"
            # self.Xpath = 1100
            self.Xpath = self.middleX + (self.middleX / 2)
            print self.Xpath
        else:
            print "turn left"
            # self.Xpath = 100
            self.Xpath = self.middleX - (self.middleX / 2)
            print self.Xpath

        CORD = (self.Xpath, self.Ypos)
        return CORD

########
# MAIN #
def main():
    # decide wich method you want to run --> 1 = disparity method, 2 = classification method
    methodDecide = 1
    isObstacleInfront_based_on_radius = False
    createdModel = False # toogle this value if you want to train the classifier
    folderName_saveImagesSuper = "superpixelImagesSaved"
    folderName_saveImages = "disparityImagesSaved"
    pairNumberSuper = 0
    pairNumber = 0
    radiusTresh = 30
    isObsticleInFrontTreshValue = 0.345 # if the value is above this. then we treat it as it is a object in front of the stereo camera
    ##### New method here that load all the images in a folder and
    #isObsticleInFrontTreshValue = 1.7  #under the sea trials. but i think i changed some code. Now it is working great again with 0.3

    #dirPath = r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulation\simulationImages1"

    #dirPathLeft = r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\images close to transponder\Left"
    #dirPathRight = r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\images close to transponder\Right"

    dirPathLeft = r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\repeatExperiment\Left"
    dirPathRight = r"C:\CV_projects\ROV_objectAvoidance_StereoVision\simulationClean\repeatExperiment\Right"

    imgsLeft = Images(dirPathLeft)
    imgsRight = Images(dirPathRight)

    imgsLeft.loadFromDirectory(dirPathLeft, False)
    imgsRight.loadFromDirectory(dirPathRight, False)

    imageListLeft = imgsLeft.getimage_list()
    imgsListRight = imgsRight.getimage_list()

    cv2.startWindowThread()

    # ---> open files to write to
    toktName = "tokt1"
    timeTXTfileName = "timeImages_" + toktName + ".txt"
    timeTXTfile = open(timeTXTfileName, 'w')
    MESSAGE_File = open("MESSAGE_File.txt", 'w')
    pathDirFile = open("pathDir.txt", 'w')

    # set up ques
    ##############
    pts_que_center = deque(maxlen=15)
    pts_que_radius = deque(maxlen=15)
    # Tresh value
    pts_que_center_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10),
                           deque(maxlen=10)]  # list holds 5 elements
    pts_que_radius_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10),
                           deque(maxlen=10)]
    yMove_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

    # If we know the size of an object, then we can calculate a good approximation to the distance we have to this object
    # Todo: be able to set this value from GUI level
    object_real_world_mm = 500  # 1000mm = 1 meter to calculate distance to a known object.

    # 1 Get or create model
    # Create segmented model
    if methodDecide == 2:
        imageOcean = cv2.imread("tokt1_R_1037.jpg")
        imageOther = cv2.imread("raptors.png")
        modelClass = modelTools(createdModel, imageOcean, imageOther)
        model = modelClass.get_model()

    # create model of the whole image
    #imageOcean = cv2.imread("tokt1_R_1037.jpg")
    #imageOther = cv2.imread("raptors.png")
    #simpleModelClass = simpleModelTools(createdModel, imageOcean, imageOther)
    #modelSimple = simpleModelClass.get_model()

    ########################################################################
    print  "starting here"
    #for i, img in enumerate(imageList[:-1]):
    for i, img in enumerate(imageListLeft[:-1]):
       # 1 get left and right images
        print "read in the images"
        frame_left = imageListLeft[i]
        frame_right = imgsListRight[i]

        ####### Run disparity method ##############
        if methodDecide == 1:
            # initialize DisparityImage class
            dispClass = DisparityImage(imgLeft=frame_left, imgRight=frame_right)
            # run the DisparityImage class process
            disparity_visual = dispClass.process()
            disparity_visualBW = cv2.convertScaleAbs(disparity_visual)

            ############### display disparity image here       #################
            cv2.imshow("disparity_visual", disparity_visual)
            cv2.waitKey(1)

            ## Save the image
            pairNumber = pairNumber + 1
            #imageName = "disparityImagesSaved2/disparity_visualBW_"+ str(pairNumber) + ".jpg"
            #cv2.imwrite(imageName, disparity_visual)
        #######################################################################

       # run superpixel method here if somthing
        elif methodDecide == 2:
            # simplePredictionClass = simplePredictionTool(frame_left, modelSimple)
            # if simplePredictionClass.get_isObstacleInFront():
            # print "object in front"

            # 2 use model to predict a new image
            # test the prediction of the model
            # image = cv2.imread("tokt1_R_267.jpg")
            print "run predictions"

            predictionClass = predictionTool(frame_left, model, radiusTresh, isObstacleInfront_based_on_radius)

            # predictionClass.show_maskedImage()
            disparity_visual = predictionClass.get_maskedImage()
            disparity_visualBW = disparity_visual
            print " done running predictions"
            cv2.imshow("disparity_visual", disparity_visual)
            cv2.waitKey(1)


            pairNumberSuper = pairNumberSuper + 1
            imgNameString_super = folderName_saveImagesSuper + "/" + toktName + "_super_map_" + str(pairNumberSuper) + ".jpg"
            cv2.imwrite(imgNameString_super, disparity_visual)

            #else:
            #    print "no object infront"
            #    continue

       ###############################################################################3
        # 2 calculate centroid info from disparity image
        # used to find out of information in the image to use for the obstacle avoidance module
        # create class
        centroidClass = centroidTools(imgBW=disparity_visual, object_real_world_mm=object_real_world_mm)
       ####### --> get the center of the object
        objectAVGCenter = centroidClass.get_objectCenter()

        ##### --> get the center of each countour for drawing class later...
        #centerCordinates = centroidClass.get_centerCordinates()

        # object_real_world_mm = 500 # 1000mm = 1 meter
        # distance_mm = proc.calcDistanceToKnownObject(self.object_real_world_mm, pixelSizeOfObject)
        # distance_mm = centroidClass.calcDistanceToKnownObject(object_real_world_mm)

        ################################################################################
        # 3 Obstacle checking the centroid information
        CORD = None
        MESSAGE=None
        try:
            obstacleClass = ObstacleAvoidance(disparity_visual= disparity_visual, isObsticleInFrontTreshValue= isObsticleInFrontTreshValue, objectAVGCenter= objectAVGCenter)
            CORD = obstacleClass.get_CORD()

            #MESSAGE = obstacleClass.process()
            MESSAGE = obstacleClass.get_MESSAGE()
            print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        except:
            pass

        ###############################################3
        # 4 send UDP mesaage with obstacle info to control system
        try:
            UDPClass = TalkUDP(MESSAGE)
            UDPClass.sendUDPmessage()
        except:
            pass

        ################ Draw stuff ##################################
        # 5 display information to the user

        ###################### draw stuff on top of disparity map #################################
        try:
            ####### make image that buffers "old" "objectCenter"
            pts_que_center_List = centroidClass.draw_bufferCenterPosition(pts_que_center)
            # get the drawn image and draw the bounding box
            drawnImage = centroidClass.drawBoundingBox()
            cv2.imshow("drawnImage", drawnImage)
            cv2.waitKey(1)

            ############## save the disparity with drawings ontop######################
            imgNameString_DISPARITY = folderName_saveImages + "/" + toktName + "_Disp_map_" + str(pairNumber) + ".jpg"
            cv2.imwrite(imgNameString_DISPARITY, drawnImage)
            #drawClass.drawTextMessage(str(distance_mm))
        except:
            pass

        ##################################################################################################################
        # 6 save information in .txt file
        print "i"
        print i

        # sends them at a choosen intervall to make the timing similar as the real world example
        # time.sleep(0.35)
        ############# writing the path direction to text file for annalysing later ##########
        if (CORD != None):
            path_string = "path direction in pixel values : " + str(CORD)
            print "path direction in pixel values : " + str(CORD)
            print "saving pathDir.txt"
            # with open("pathDir.txt", 'w') as f:
            pathDirFile.write(path_string + '\n')

        if (MESSAGE != None):
            MESSAGE_File.write(MESSAGE + '\n')

        # write the time these images have been taken to a file
        dateTime_string = unicode(datetime.datetime.now())
        pairNumber = i
        path_string = str(pairNumber) + " , " + str(dateTime_string)
        print "saving timeImages.txt"
        timeTXTfile.write(path_string + '\n')
        print "done saving timeImages.txt"

    # close the .txt files that had been written to
    try:
        pathDirFile.close()
        MESSAGE_File.close()
        timeTXTfile.close()
    except:
        pass

if __name__ == '__main__':
    main()