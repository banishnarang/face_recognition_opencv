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
    face_haar = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    # Extracting the face from the image
    face = face_haar.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=3)

    return face, gray_img


def label_training_data(directory):
    '''
    This function takes in a directory of images (one directory per person),
    Extracts the face using face_detection function for each image,
    Returns a list of faces extracted from all the images
    and a list of their corresponding ids.
    '''

    face_list = []
    face_id_list = []

    for path, sub_dir_name, files in os.walk(directory):
        for filename in files:

            # if the name starts with period, it's a system file, so we skip it
            if filename.startswith('.'):
                print('Skipping System File')
                continue

            # Save the name as id and path as the img_path
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)

            # Printing the image path and id
            print('img_path: ', img_path)
            print('id: ', id)

            # Load and save the image
            train_img = cv2.imread(img_path)

            # If image directory is empty, print message and skip it
            if train_img is None:
                print('No Images Found!')
                continue

            # Call face_detection function and get extracted face and grayscale img
            face, gray_img = face_detection(train_img)

            # Skip to the next one if no face is extracted from the image
            if len(face) != 1:
                continue

            # x-axis, y-axis, width, height
            (x, y, w, h) = face[0]

            # Extracting the Reason of Interest from the grayscale image
            roi_gray = gray_img[x:x+w, y:y+h]

            face_list.append(roi_gray)
            face_id_list.append(int(id))

        return face_list, face_id_list


def train_classifier(face_list, face_id_list):
    '''
    This function trains the face classfier using LBPH
    takes face and face id as input
    and returns the classfier
    '''

    face_clf = cv2.face.LBPHFaceRecognizer_create()
    face_clf.train(face_list, np.array(face_id_list))

    return face_clf


def draw_rect(test_img, face):
    '''
    This function draws a rectangle around the face
    while classifying the face
    '''

    (x, y, w, h) = face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (0, 255, 0), thickness=3)


def write_text(test_img, text, x, y):
    '''
    This function is used to write the label upon classification on the image
    :param test_img:  The image being classified
    :param text:      The label text to be written
    :param x:         x coordinate position
    :param y:         y coordinate position
    :return:          None
    '''
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX, 3, (255, 0, 0), 6)