import argparse

import cv2 as cv
import face_recognition
import numpy as np

cli = argparse.ArgumentParser()
cli.add_argument("-i", help="path to image file", required=True)
cli.add_argument("-o", help="path to save faces", required=True)
cli.add_argument("--type",type=int, required=True)

args = cli.parse_args()

path_to_save = args.o
type = args.type
image_path = args.i

image = face_recognition.load_image_file(image_path)
face_landmarks_list = face_recognition.face_landmarks(image)
image = cv.imread(image_path)
if type == 1:
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            if facial_feature in ["left_eyebrow", "right_eyebrow", "top_lip", "bottom_lip", "left_eye", "right_eye","chin","nose_bridge","nose_tip"]:
                cv.polylines(image, np.int32([face_landmarks[facial_feature]]), False, (255,255,255), 1)
    cv.imwrite(path_to_save, image)
elif type == 2:
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            if facial_feature == "left_eyebrow" or facial_feature == "right_eyebrow":
                mask = image.copy()
                cv.fillPoly(mask,np.int32([face_landmarks[facial_feature]]),(50,70,100))
                cv.addWeighted(image, 0.5, mask, 0.5, 0, image)
                cv.polylines(image, np.int32([face_landmarks[facial_feature]]), False, (60, 80, 100), 2)
            elif facial_feature == "top_lip" or facial_feature == "bottom_lip":
                mask = image.copy()
                cv.fillPoly(mask,np.int32([face_landmarks[facial_feature]]),(0,0,153))
                cv.addWeighted(image, 0.5, mask, 0.5, 0, image)
                cv.polylines(image, np.int32([face_landmarks[facial_feature]]), False, (0,0,180), 2)
            elif facial_feature == "left_eye" or facial_feature == "right_eye":
                mask = image.copy()
                cv.polylines(mask,np.int32([face_landmarks[facial_feature]]),True,(12,12,12),3)
                cv.addWeighted(image, 0.5, mask, 0.5, 0, image)
    cv.imwrite(path_to_save, image)
