#python justtracking.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

import imutils
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

class_to_track = 'person'

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

num_frames = 0
person_detected = False
person_bbox = None
tracker = None

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    num_frames += 1

    if not person_detected:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # Class index of 'person'
                    person_detected = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    person_bbox = (box.astype("int"))
                    (startX, startY, endX, endY) = person_bbox
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, person_bbox)
                    break

    if person_detected and tracker:
        ok, person_bbox = tracker.update(frame)
        if ok:
            (startX, startY, w, h) = [int(v) for v in person_bbox]
            cv2.rectangle(frame, (startX, startY), (startX + w, startY + h), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()

end_time = time.time()
elapsed_time = end_time - start_time
fps_val = num_frames / elapsed_time

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
print("[INFO] Secondary approx. FPS: {:.2f}".format(fps_val))

cv2.destroyAllWindows()
vs.stop()

