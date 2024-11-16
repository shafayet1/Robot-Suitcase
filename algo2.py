import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('guy.jpg',cv2.IMREAD_COLOR)


low_cascade = cv2.CascadeClassifier('haarcascade_lowerbody.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )

low = low_cascade.detectMultiScale(gray, 1.1 , 3)
    
for (x,y,w,h) in low:
    cv2.rectangle(img, (x,y), (x+w, y+h), (12,150,100),2)
    
cv2.imshow('image',img)
cv2.waitKey(0) # If you don'tput this line,thenthe image windowis just a flash. If you put any number other than 0, the same happens.
cv2.destroyAllWindows()