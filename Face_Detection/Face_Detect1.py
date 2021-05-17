# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:50:19 2021

@author: Kleogis
"""
import cv2

image = cv2.imread('people1.jpg')
print(image.shape)

#cv2.imshow(image)
#cv2.imshow('ImageWindow', image)


image = cv2.resize(image, (800, 600))
print(image.shape)

cv2.startWindowThread()
cv2.namedWindow("preview")
cv2.imshow("preview", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("preview", image_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detections = face_detector.detectMultiScale(image_gray, scaleFactor = 1.09)
for (x, y, w, h) in detections:
  #print(x, y, w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0,255,255), 5)
cv2.imshow("preview", image)
cv2.waitKey(0)
cv2.destroyAllWindows()