# This script runs the LBP method in a simulation
# To run: type this in your terminal in the correct folder
# python simulation_haralick.py

# Change the folder you want to run the sequence of images on by changing (Inside the main() method)
#dirPathLeft = r"repeatExperiment\Left"
#dirPathRight = r"repeatExperiment\Right"


import numpy as np
import cv2

import socket # to send UDP message to labview pc

from collections import deque # to keep track of a que of last poitions of the center of an object
import datetime # to print the time to the textfile

from importDataset import Images  # Import the functions made in importDataset

from imageTools import centroidTools

# import the classes to do the superpixelMethod
from slicSuperpixel_Haralick_method import modelToolsH
from slicSuperpixel_Haralick_method import predictionToolH

#################################################
class TalkUDP(object):
    # Comunication methods

    def __init__(self, MESSAGE) :#, disparity_visual):
        #self.MESSAGE = ObstacleAvoidance.getMessage()
        self.MESSAGE = MESSAGE

    def sendUDPmessage(self):

        # addressing information of target
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

class ObstacleAvoidance(object):

    def __init__(self, disparity_visual, isObsticleInFrontTreshValue, objectAVGCenter, center):
        self.disparity_visual = disparity_visual
        self.isObsticleInFrontTreshValue = isObsticleInFrontTreshValue
        self.objectAVGCenter = objectAVGCenter
        self.cx, self.cy = self.objectAVGCenter
        self.center = center

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
        # cv2.waitKey(0)
        # print meanValue
        # cv2.waitKey(0)
        # if the meanValue is above a treshold for to "small areas of pixels in the image"
        # in this case 0.3
        #if self.meanValue > self.isObsticleInFrontTreshValue:  # 1.7:
        if self.meanValue < self.isObsticleInFrontTreshValue:  # 1.7:
            return True
        else:
            return False

    def createMESSAGE(self):
        # directionMessage = "status : "
        # directionMessage = "status : , " # todo : uncoment it later

        directionMessage = "status : "
        #####
        # --> tell path program
        # 0 if there is obstacle in the image
        # 1 if there is NO obstacle in the image
        if self.isObsticleInFront():  # if the treshold says there is somthing infront then change directions
            # it should change path
            directionMessage = directionMessage + str(0) + " "
        else:  # if nothing is in front of camera, do not interupt the path
            # it can continue on its path
            directionMessage = directionMessage + str(1) + " "

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

    #dirPathLeft = r"images close to transponder\Left"
    #dirPathRight = r"images close to transponder\Right"

    dirPathLeft = r"repeatExperiment\Left"
    dirPathRight = r"repeatExperiment\Right"

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
    yMove_list = [deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), deque(maxlen=10)]

    # If we know the size of an object, then we can calculate a good approximation to the distance we have to this object
    object_real_world_mm = 500  # 1000mm = 1 meter to calculate distance to a known object.

    # 1 Get or create model
    # Create segmented model
    imageOcean = cv2.imread("tokt1_R_1037.jpg")
    imageOther = cv2.imread("raptors.png")
    modelClass = modelToolsH(createdModel, imageOcean, imageOther)
    model = modelClass.get_model()

    ########################################################################
    print  "starting here"
    for i, img in enumerate(imageListLeft[:-1]):
       # 1 get left and right images
        print "read in the images"
        frame_left = imageListLeft[i]
        frame_right = imgsListRight[i]

       # Display the original image
        cv2.imshow("left camera image", frame_left)
        cv2.waitKey(1)

       # run superpixel method here if somthing
        # 2 use model to predict a new image
        print "run predictions"

        predictionClass = predictionToolH(frame_left, model, radiusTresh, isObstacleInfront_based_on_radius)

        disparity_visual = predictionClass.get_maskedImage()
        print " done running predictions"
        cv2.imshow("disparity_visual", disparity_visual)
        cv2.waitKey(1)

        pairNumberSuper = pairNumberSuper + 1
        imgNameString_super = folderName_saveImagesSuper + "/" + toktName + "_super_map_" + str(pairNumberSuper) + ".jpg"
        cv2.imwrite(imgNameString_super, disparity_visual)

       ###############################################################################3
        # 2 calculate Centroid info from disparity image
        # used to find out of information in the image to use for the obstacle avoidance module
        # create class
        centroidClass = centroidTools(imgBW=disparity_visual, object_real_world_mm=object_real_world_mm)
        ####### --> get the center of the object
        objectAVGCenter = centroidClass.get_objectCenter()

        ################################################################################
        # 3 Obstacle checking the centroid information
        CORD = None
        MESSAGE=None
        try:
            obstacleClass = ObstacleAvoidance(disparity_visual= disparity_visual, isObsticleInFrontTreshValue= isObsticleInFrontTreshValue, objectAVGCenter= objectAVGCenter, center= center)
            CORD = obstacleClass.get_CORD()

            MESSAGE = obstacleClass.get_MESSAGE()
            print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
        except:
            pass

        ###############################################3
        # 4 Send UDP mesaage with obstacle info to control system
        try:
            UDPClass = TalkUDP(MESSAGE)
            UDPClass.sendUDPmessage()
        except:
            pass

        ###################### DRAW information on top of disparity map #################################
        # 5 display information to the user
        try:
            ####### make image that buffers "old" "objectCenter"
            pts_que_center_List = centroidClass.draw_bufferCenterPosition(pts_que_center)
            # get the drawn image and draw the bounding box
            drawnImage = centroidClass.drawBoundingBox()
            cv2.imshow("drawnImage", drawnImage)
            cv2.waitKey(1)

            ############## Save the disparity with drawings ontop######################
            # 5.5 save the image created
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