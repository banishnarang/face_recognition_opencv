import numpy as np
import cv2
import os


def face_detection(img):
    '''
      Function to detect the face from the image.
      It converts the image into grayscale and then extracts the face.
      Returns the extracted face and converted grayscale image.
    '''

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Using Haar Cascade Classifier from opencv repo
    face_haar = cv2.CascadeClassifier(
        'https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml')

    # Extracting the face from the image
    face = face_haar.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)

    return face, gray_img