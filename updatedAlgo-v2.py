# Run This Commmand 
#python updatedAlgo.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import imutils
# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import dlib
import time
import cv2


# from deepsort_tracker import DeepSort 

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

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

trackers = []
labels = []

# initialize the video stream, allow the camera sensor to warm up,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
        0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > args["confidence"]:

            idx = int(detections[0, 0, i, 1])
            
            # only look at people
            if CLASSES[idx] != class_to_track: 
                continue
        
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # tracker
            t = dlib.correlation_tracker()
            rect = dlib.rectangle (startX, startY, endX, endY)
            t.start_track(rgb, rect)
            trackers.append(t)

            # draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
                confidence * 100)
            labels.append(label)

            # cv2.rectangle(frame, (startX, startY), (endX, endY),
            #     COLORS[idx], 2)
            # y = startY - 15 if startY - 15 > 15 else startY + 15
            # cv2.putText(frame, label, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                
        for (t, l) in zip(trackers, labels): 
            t.update(rgb)
            pos = t.get_position()
            startX = int (pos.left())
            startY = int (pos.top())
            endX = int (pos.right())
            endY = int (pos.bottom())
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
            cv2.putText(frame, label, (startX, startY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            
            index = trackers.index(t)
            cv2.putText(frame, "ID: " + str(index), (startX, startY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
