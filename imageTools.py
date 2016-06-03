import cv2
import numpy as np

class centroidTools(object):

    def __init__(self, imgBW, object_real_world_mm):
        # self.MESSAGE = ObstacleAvoidance.getMessage()
        self.imgBW = self.prepareDisparityImage_for_centroid(imgBW) # BW for black and white
        self.object_real_world_mm = object_real_world_mm

        self.objectCenter = (0,0)

        self.focallength_mm = (2222.72426 * 35) / 1360
        self.pxPERmm = 2222.72426 / self.focallength_mm  # pxPERmm = 38.8571428572
        self.pixelSizeOfObject = 50

        # calculate the centers of the small "objects"
        self.imgBWCopy, self.centerCordinates = self.findCentroidsCenterCords()
        try:
            # calculate the average center of this disparity
            self.objectAVGCenter = self.getAverageCentroidPosition()
        except:
            pass

        #centerCordinates = self.get_centerCordinates()

        self.distance_mm = self.calcDistanceToKnownObject()

    def get_imgBWCopy(self):
        return self.imgBWCopy


    def findCentroidsCenterCords(self):
        # calculate the centers of the small "objects"
        imgBWCopy = self.imgBW.astype(np.uint8)

        contours0, hierarchy = cv2.findContours( imgBWCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        moments = [cv2.moments(cnt) for cnt in contours0]

        # rounded the centroids to integer.
        centroids = [( int(round(m['m10']/m['m00'])), int(round(m['m01']/m['m00'])) ) for m in moments]

        centerCordinates = []

        for ctr in centroids:
            # draw a black little empty circle in the centroid position
            centerCircle_Color = (0, 0, 0)
            #cv2.circle(self.imgBW, ctr, 4, centerCircle_Color)
            cv2.circle(imgBWCopy,ctr,4,centerCircle_Color)
            centerCordinates.append(ctr)


        centerCordinates = np.asarray(centerCordinates)

        return imgBWCopy, centerCordinates   # self.imgBW, self.centerCordinates

    def get_centerCordinates(self):
        return self.centerCordinates

    def getAverageCentroidPosition(self):
        # taking the average of the centroids x and y poition to calculate and estimated  object CENTER
        # centerCordinates = centerCordinates.astype(np.int)
        objectCenter = np.average(self.centerCordinates, axis=0)
        # objectCenter = objectCenter.astype(np.uint8)

        print "objectCenter  : "
        print objectCenter  # , objectCentery
        # print objectCentery

        # Unpack tuple.
        (objectCenterX, objectCenterY) = self.objectCenter

        # Display unpacked variables.
        print(objectCenterX)
        print(objectCenterY)
        # pack the tupple
        objectCenter = (int(objectCenterX), int(objectCenterY))

        return objectCenter

    def calcDistanceToKnownObject(self):  #object_real_world_mm):
        object_image_sensor_mm = self.pixelSizeOfObject / self.pxPERmm
        #distance_mm = (object_real_world_mm * self.focallength_mm) / object_image_sensor_mm
        distance_mm = (self.object_real_world_mm * self.focallength_mm) / object_image_sensor_mm
        return distance_mm

    def prepareDisparityImage_for_centroid(self, IMGbw):
        # This method dilates the points in the black and white image.
        # This is to connect more of the dots, so we get bigger countours.

        # DILATE white points...
        IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
        IMGbw = cv2.dilate(IMGbw, np.ones((5, 5)))
        return IMGbw

    # object avoidance      #################
    def findBiggestObject(self, imgBW, pts_que_center, pts_que_radius, radiusTresh=40):

        blurred = cv2.GaussianBlur(imgBW, (11, 11), 0)
        mask = blurred
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        biggestObjectCenter = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            biggestObjectCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > radiusTresh:  # works as a treshold
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(imgBW, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(imgBW, biggestObjectCenter, 5, (0, 0, 255), -1)

        # update the points queue
        try:
            pts_que_center.appendleft(biggestObjectCenter)
            pts_que_radius.appendleft(radius)
            pts_que_center_List = list(pts_que_center)
            pts_que_radius_List = list(pts_que_radius)
        except:
            pass

        # loop over the set of tracked points
        for i in xrange(1, len(pts_que_center)):
            # if either of the tracked points are None, ignore
            # them
            if pts_que_center[i - 1] is None or pts_que_center[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(15 / float(i + 1)) * 2.5)
            cv2.line(imgBW, pts_que_center[i - 1], pts_que_center[i], (255, 255, 255), thickness)

        return imgBW, biggestObjectCenter, pts_que_center_List, pts_que_radius_List

    def process(self):
        #centroidClass = centroidTools(imgBW=disparity_visual)
        # calculate the centers of the small "objects"
        #self.imgBW, self.centerCordinates = self.findCentroidsCenterCords()
        # pixelSizeOfObject = 50
        # calculate the average center of this disparity
        #try:
            # objectAVGCenter = proc.getAverageCentroidPosition(centerCordinates)
        #    objectAVGCenter = self.getAverageCentroidPosition()
        #except:
        #    pass
            # isObstacleInfront_based_on_radius = False

        #centerCordinates = self.get_centerCordinates()

        # object_real_world_mm = 500 # 1000mm = 1 meter
        # distance_mm = proc.calcDistanceToKnownObject(self.object_real_world_mm, pixelSizeOfObject)
        distance_mm = self.calcDistanceToKnownObject()

        # update radiusTresh with tuner
        # radiusTresh = cv2.getTrackbarPos('radiusTresh', trackBarWindowName)
        ####### make image that buffers "old" centerpoints, and calculate center of the biggest centroid -- hopefully that is the biggest object
        try:
            imgStaaker, center, pts_que_center_List, pts_que_radius_List = centroidClass.findBiggestObject(
                disparity_visualBW.copy(), pts_que_center, pts_que_radius, radiusTresh=radiusTresh)
        except:
            pass

class drawTools(object):
    def __init__(self, image, centerCordinates, objectAVGCenter):
        self.image = image
        self.centerCordinates = centerCordinates
        self.objectAVGCenter = objectAVGCenter


    def drawPath(self, Xpath, Ypos, image):
        radius = 100
        # (x,y) = centerCordinates
        # center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(image, (Xpath, Ypos), radius, (255, 255, 255), 7)

        return image

    def drawBox(self):
        ############## creating a minimum rectangle around the object ######################
        try:
            rect = cv2.minAreaRect(points=self.centerCordinates)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (255, 255, 255), 2)
        except:
            pass

    def circle_around_object(self):
        ########### circle around object #######3
        # need try since: OpenCV Error: Assertion failed (points.checkVector(2) >= 0 && (points.depth() == CV_32F || points.depth() == CV_32S)) in cv::minEnclosingCircle,
        # if less then 2 "points" in image the mehod has problem making a circle...
        try:
            (x, y), radius = cv2.minEnclosingCircle(self.centerCordinates)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(self.image, center, radius, (255, 255, 255), 2)
        except:
            pass

    def elipse_around_object(self):
        ########### finding a elipse ##############
        # if len(centerCordinates) > 5:  # need more than points than 5 to be able to run  cv2.fitEllipse
        try:
            ellipse = cv2.fitEllipse(self.centerCordinates)
            cv2.ellipse(self.image, ellipse, (255, 255, 255), 2)
        except:
            pass

    def fitting_line_thruObject(self):
        ##### fitting a line ###########
        try:
            rows, cols = self.image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(points=self.centerCordinates, distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01,
                                         aeps=0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(self.image, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
        except:
            pass

    # drawTextMessage(image_color_with_Draw, str(distance_mm))
    def drawTextMessage(self, text):
        try:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image, text, (10, 500), font, 1, (255, 255, 255), 2)

        except:
            pass

    def drawAVGcenter_circle(self):
        # draw the new objectAVGCenter in a white circle
        centerCircle_Color = (255, 255, 255)
        cv2.circle(self.image, self.objectAVGCenter, 10, centerCircle_Color)

    def get_drawnImage(self):
        return self.image

    def saveImage(self, image_name_str, image):
        cv2.imwrite(image_name_str, image)

    def drawStuff(self, centerCordinates):
        # http://opencvpython.blogspot.no/2012/06/contours-2-brotherhood.html
        # http://docs.opencv.org/3.1.0/dd/d49/tutorial_py_contour_features.html#gsc.tab=0pyth
        ############## creating a minimum rectangle around the object ######################
        try:
            rect = cv2.minAreaRect(points=centerCordinates)
            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (255, 255, 255), 2)
        except:
            pass
        ########### circle around object #######3

        try:
            (x, y), radius = cv2.minEnclosingCircle(centerCordinates)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(self.image, center, radius, (255, 255, 255), 2)
        except:
            pass

        ########### finding a elipse ##############
        # if len(centerCordinates) > 5:  # need more than points than 5 to be able to run  cv2.fitEllipse
        try:
            ellipse = cv2.fitEllipse(centerCordinates)
            cv2.ellipse(self.image, ellipse, (255, 255, 255), 2)
        except:
            pass

        ##### fitting a line ###########

        try:
            rows, cols = self.image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(points=centerCordinates, distType=cv2.cv.CV_DIST_L2, param=0, reps=0.01,
                                         aeps=0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(self.image, (cols - 1, righty), (0, lefty), (255, 255, 255), 2)
        except:
            pass

        try:
            pixelSizeOfObject = radius  # an okay estimate for testing
        except:
            pixelSizeOfObject = 50

        return self.image, pixelSizeOfObject
