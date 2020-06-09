import numpy as np
import cv2
import os

import face_recognition as fr

# Just to check the path of the py file
print(fr)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Read/Load from the saved model
face_recognizer.read(r'model\face_recognition_0.yml')

capture = cv2.VideoCapture(0)  # Replace 0 with a path if you want to use a video on local disk

# Assign labels for faces in the dictionary 'name' as (key, value) pairs
name = {0: 'Banish'}   # 1, 2.... for adding more faces

while True:

    ret, test_img = capture.read()
    
    faces_detected, gray_img = fr.face_detection(test_img)
    
    #print("Face Detected: ", faces_detected)
    
    for (x, y, w, h) in faces_detected:
        
        cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), thickness=5)

    for face in faces_detected:
        (x, y, w, h) = face
    
        # Extract ROI
        roi_gray = gray_img[y:y+w, x:x+h]
    
        # Name of the person and confidence in the prediction
        label, confidence = face_recognizer.predict(roi_gray)
    
        print('Confidence: ', confidence)
        print('Label: ', label)
    
        # Drawing rectangle on the face using draw_rect func
        fr.draw_rect(test_img, face)
    
        # Save the name as predict_name
        predict_name = name[label]
    
        # Lower the confidence, better the results (threshold)
        if confidence > 90:
            fr.write_text(test_img, 'Unknown', x, y)
            continue
    
        # Add the label/name to the image using write_text func
        fr.write_text(test_img, predict_name, x, y)

    # Let's resize the image for display
    resize_img = cv2.resize(test_img, (1000, 700))

    # Display the image
    cv2.imshow('Face Recognition', resize_img)

    if cv2.waitKey(10) == ord('q'):
        break