import cv2
import numpy as np

class centroidTools(object):

    def __init__(self, imgBW, object_real_world_mm):
        self.imgBW = imgBW
        self.object_real_world_mm = object_real_world_mm

        self.objectCenter = (0,0)
        self.focallength_mm = (2222.72426 * 35) / 1360
        self.pxPERmm = 2222.72426 / self.focallength_mm  # pxPERmm = 38.8571428572
        self.pixelSizeOfObject = 50

        # calculate the centers of the small "objects"
        #self.objectCenter = self.findCentroidsCenterCords()
        try:
            # calculate the average center of this disparity
            # calculate the centers of the small "objects"
            self.objectCenter, self.drawImage, self.contours0 = self.findObjectCenter_and_CentroidsCenterCords()
        except:
            pass

        self.distance_mm = self.calcDistanceToKnownObject()

    def findObjectCenter_and_CentroidsCenterCords(self):
        # calculate the centers of the small "objects"

        # The image need to be monochrome i.e only one channel --> no (r,g,b)
        # to be able to run findContours
        image = self.imgBW.astype(np.uint8)
        print image.dtype
        #imageMono = cv2.cvtColor(self.imgBW, cv2.cv.CV_BGR2GRAY)

        # prepare image for centroid calculations
        # This method dilates the points in the black and white image.
        # This is to connect more of the dots, so we get bigger countours.
        # DILATE white points...
        image = cv2.dilate(image, np.ones((5, 5)))

        # cv2.CHAIN_APPROX_SIMPLE --> returns not all points of coutours as "cv2.CHAIN_APPROX_NONE" and is therefore faster and takes less memory
        # cv2.RETR_EXTERNAL --> only returns the external countours --> faster
        (contours0, _) = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroid_XList = []
        centroid_YList = []

        areaTot = 0
        for (i, cnt) in enumerate(contours0):
            area = cv2.contourArea(cnt)

            # compute the moments of the contour
            # use the moments to compute the "center of mass" of each contour
            m = cv2.moments(cnt)
            centroid_X = [area * int(round(m['m10'] / m['m00']))]
            centroid_y = [area * int(round(m['m01'] / m['m00']))]

            centroid_XList.append(centroid_X)
            centroid_YList.append(centroid_y)

            areaTot = areaTot + area

        centroid_XList = np.asarray(centroid_XList)
        centroid_YList = np.asarray(centroid_YList)

        # take the average
        centroid_XListCenters = centroid_XList / areaTot
        centroid_YListCenters = centroid_YList / areaTot

        # sum the points and cast to int so cv2.draw works
        objectCenterX = int(np.sum(centroid_XListCenters))
        objectCenterY = int(np.sum(centroid_YListCenters))

        print objectCenterX
        print objectCenterY

        #### For vizualizationg the countours ############
        cv2.drawContours(image, contours0, -1, (255, 255, 255), 2)
        print "Found {} contours".format(len(contours0))

        # draw the center of the object on the image
        cv2.circle(image, (objectCenterX, objectCenterY), 10, (255, 255, 255), 2)

        # show the output image
        #cv2.imshow("All Contours", image)
        #cv2.waitKey(1)
        ###################################

        objectCenter = (objectCenterX, objectCenterY)

        return objectCenter, image, contours0

    def drawBoundingBox(self):
        # Make a bounding box with some margin around the obstacle
        xLast, yLast = self.drawImage.shape[:2]
        wLast = 0
        hLast = 0

        for c in self.contours0:
            # fit a bounding box to the contour
            (x, y, w, h) = cv2.boundingRect(c)

            if (xLast > x):
                xLast = x

            if (yLast > y):
                yLast = y

            # if (wLast < w):
            wLast = wLast + w

            # if (hLast < h):
            hLast = hLast + h

        cv2.rectangle(self.drawImage, (xLast, yLast), (xLast + wLast, yLast + hLast), (255, 255, 255), 2)

        # Display the drawn image
        #cv2.imshow("drawnImage", self.drawImage)
        #cv2.waitKey(1)
        return self.drawImage

    def get_objectCenter(self):
        return self.objectCenter

    def get_centerCordinates(self):
        return self.centerCordinates

    def draw_bufferCenterPosition(self, pts_que_center):
        # method that uses a que to draw a line to
        # shows how the center has moved in the image over time

        # update the points queue
        try:
            pts_que_center.appendleft(self.objectCenter)
            pts_que_center_List = list(pts_que_center)
            #pts_que_radius.appendleft(radius)

            #pts_que_radius_List = list(pts_que_radius)
        except:
            pass

        # loop over the set of tracked points
        for i in xrange(1, len(pts_que_center)):
            # if either of the tracked points are None, ignore
            # them
            if pts_que_center[i - 1] is None or pts_que_center[i] is None:
                continue

            # otherwise, compute the thickness of the line and

            thickness = int(np.sqrt(15 / float(i + 1)) * 2.5)
            # draw the connecting lines
            cv2.line(self.drawImage, pts_que_center[i - 1], pts_que_center[i], (255, 255, 255), thickness)

        return pts_que_center_List #, pts_que_radius_List

    def calcDistanceToKnownObject(self):  #object_real_world_mm):
        object_image_sensor_mm = self.pixelSizeOfObject / self.pxPERmm
        distance_mm = (self.object_real_world_mm * self.focallength_mm) / object_image_sensor_mm
        return distance_mm

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

# drawTools is not used, as drawing has been moved into centroidTools
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

