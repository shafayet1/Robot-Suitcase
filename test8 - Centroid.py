# Run This Command
# python test8.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
from picamera2 import Picamera2
from time import sleep

from scipy.spatial import distance as dist
from collections import OrderedDict

# from deepsort_tracker import DeepSort 
from picarx import Picarx

# Centroid tracking class definition
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
    
# PI CAR 

def follow(px, xcen, startX, endX):
   #print("HELLO")
   #print(xcen)
    if (xcen < 150):
        px.set_dir_servo_angle(-20)
        px.forward(15)
        print("Forward xcen=", xcen)
        time.sleep(1)
        
        #set it back to center
        px.set_dir_servo_angle(-30)
        px.forward(15)
        #px.forward(0)
        print("Left: xcen =", xcen)
    elif (xcen >= 150):
        
        px.set_dir_servo_angle(5)
        px.forward(15)
        print("Right: xcen =", xcen)
    elif startX < 100 and endX > 320:
        #px.forward(0)
        print("Object chase complete!!")
    elif xcen == 0:
        px.forward(0)

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class_to_track = 'person'
px = Picarx()
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

xcen = 0

print("[INFO] starting video stream...")

time.sleep(2.0)
fps = FPS().start()
counter = 0

# Initialize centroid tracker
ct = CentroidTracker()

if __name__ == "__main__":
    try:
        px = Picarx()
        px.set_cam_tilt_angle(30)
        px.set_cam_pan_angle(10)
        px.set_dir_servo_angle(-20)

        cached_centroid = 0
# Loop over frames from the video stream

        while True:
            # Read a frame from the video stream
            im = picam2.capture_array()
            frame = imutils.resize(im, width=400)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                            0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            rects = []
            min_distance = float('inf')
            check = 0 

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    #check = 1
                    if idx == 15:  # Class index of 'person'
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        check = 1
                        person_bbox = (box.astype("int"))
                        (startX, startY, endX, endY) = person_bbox
                        new_centroid = (endX + startX)/2
                        distance = abs(cached_centroid - new_centroid)
                        if distance < min_distance:
                            min_distance = distance
                            rects = []
                            rects.append(person_bbox)
                    else:
                        check = 0
                                   

            if check == 1: 
                objects = ct.update(rects)

                # Loop over the tracked objects
                for (objectID, centroid) in objects.items():
                    # Draw the object ID and centroid on the frame
                    text = "ID {}".format(objectID)
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

                
                cached_centroid = centroid[0]
                (startX, startY, endX, endY) = rects[0]
                follow(px, centroid[0], startX, endX)
            else:
                px.forward(0)
                print('done moving')

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
    finally:
        px.set_cam_tilt_angle(30)
        px.set_cam_pan_angle(10)
        #px.set_dir_servo_angle(0)
        px.stop()
        sleep(.2)
