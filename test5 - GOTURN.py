# Run This Command
# python test5.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import time
import cv2
from picamera2 import Picamera2
from time import sleep

# from deepsort_tracker import DeepSort 
from picarx import Picarx

def follow(px, xcen, startX, endX):

    print(xcen)
    if (xcen < 140):
        px.set_dir_servo_angle(0)
        px.forward(15)
        print("Forward xcen=", xcen)
        time.sleep(2)
        
        #set it back to center
        px.set_dir_servo_angle(-35)
        px.forward(15)
    
        print("Left: xcen =", xcen)
    elif (xcen >= 160):
        
        px.set_dir_servo_angle(35)
        px.forward(15)
        print("Right: xcen =", xcen)
    elif (xcen >= 140 and xcen < 160):
        px.set_dir_servo_angle(0)
        px.forward(15)
        print("Forward xcen=", xcen)
        time.sleep(2)
    elif startX < 100 and endX > 320:
    
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

if __name__ == "__main__":
    try:
        px = Picarx()
        px.set_cam_tilt_angle(25)
        px.set_cam_pan_angle(10)
        px.set_dir_servo_angle(-20)

        while True:
            im = picam2.capture_array()
            frame = imutils.resize(im, width=400)

            (h, w) = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                0.007843, (300, 300), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    idx = int(detections[0, 0, i, 1])
                    if CLASSES[idx] != class_to_track:
                        continue
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    xcen = (endX + startX) / 2
                    check = 1
                    print(xcen)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    break
                else:
                    check = 0
                    xcen = 0

            cv2.imshow("Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if check == 1:
                follow(px, xcen, startX, endX)
            else:
                px.forward(0)
                print('done moving')

            if key == ord("q"):
                break

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # do a bit of cleanup
        cv2.destroyAllWindows()
    finally:
        px.set_cam_tilt_angle(25)
        px.set_cam_pan_angle(10)
        #px.set_dir_servo_angle(0)
        px.stop()
        sleep(.2)
