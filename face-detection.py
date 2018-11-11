# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 02:14:56 2018

@author: Mahbub

"""

import cv2


# Import haarCascade xml: see github page for other lists of haarcascade xml
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")


# Detect
def detect(gray, frame):
    
    # get upper left co-ordinate of the face, width, height of the rectangle
    # 1.3 = the size of the images will be reduced 1.3 times
    # 5 minimum number of neighbors
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    
    
    # iterate through faces and detect eyes
    # (x, y) = co-ordinate of upper left corner
    # x+w = lower right
    # y+h = lower right
    # lower right co-ordinate (x+w, y+h)
    # w = width of rectangle
    # h = height of the rectangle
    # (255, 0, 0) color
    # 2 = thickness of the edgens of rectangle
    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        # detect eyes in a face using roi_gray
        # 1.1 = experimental value
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3) 

        # detect individual eyes
        for(ex, ey, ew, eh) in eyes:
            
            # print rectangle in a colored roi
            # 0, 255, 0 = green
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
    # return original face with colored rectangle
    return frame 


# Capture the video
# 0 = built in webcam in a computer    
# 1 = external webcam
video_capture = cv2.VideoCapture(1)

# Loop through the webcam streaming
while True:
    
    # read method returns two elements
    # we only need the frame, 
    # will use under score so we won't get the first element
    _, frame = video_capture.read()
    
    # we need gray frame
    # convert the colored frame to gray frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    # apply the detect function
    canvas = detect(gray, frame)
    cv2.imshow("Video", canvas)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# turn off the webcam
video_capture.release()
cv2.destroyAllWindows()
        
    

        
        
    
    
