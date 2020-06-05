import numpy as np
import cv2
import os

import face_recognition as fr

# Just to check the path of the py file
print(fr)

# Add path of the image you want to test the model on
test_img = cv2.imread("profile_2.jpg")

faces_detected, gray_img = fr.face_detection(test_img)
print('Face Detected: ', faces_detected)


# ----------------- Training ----------------------

# Call and extract the face_list and face_id_list from label_training_data func
face_list, face_id_list = fr.label_training_data(r'images\00')

# Call and train face recognizer/classifier from train_classifier func
face_clf = fr.train_classifier(face_list, face_id_list)

# Save the classifier/recognizer
face_clf.save(r'model\face_recognition_0.yml')

# Assign labels for faces in the dictionary 'name' as (key, value) pairs
name = {00: 'Banish'}   # 01, 02.... for adding more faces

for face in faces_detected:
    (x, y, w, h) = face

    # Extract ROI
    roi_gray = gray_img[x:x+w, y:y+h]

    # Name of the person and confidence in the prediction
    label, confidence = face_clf.predict(roi_gray)

    print('Confidence: ', confidence)
    print('Label: ', label)

    # Drawing rectangle on the face using draw_rect func
    fr.draw_rect(test_img, face)

    # Save the name as predict_name
    predict_name = name[label]

    # Add the label/name to the image using write_text func
    fr.write_text(test_img, predict_name, x, y)


# Let's resize the image for display
resize_img = cv2.resize(test_img, (1000, 700))

# Display the image
cv2.imshow('Face Recognition', resize_img)

# Close it upon user clicking 0
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------------------------