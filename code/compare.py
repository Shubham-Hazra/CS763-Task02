import argparse
import pickle

import cv2 as cv
import face_recognition

cli = argparse.ArgumentParser()
cli.add_argument("--data", help="path to image file", required=True)
args = cli.parse_args()

image_path = args.data

image_name = image_path.split("/")[-1].split(".")[0]

#Ground-Truth data
ground_truth_pickle = open("../data/captured/"+image_name+".txt", "rb")
ground_truth = pickle.load(ground_truth_pickle)

# Perform Viola-Jones face detection
image = cv.imread(image_path)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml')
faces = face_cascade.detectMultiScale(gray)

for face in faces:
    face[2] += face[0]
    face[3] += face[1]

f = open("../results/faceDetection/vj"+image_name+".txt", "w")
f.write(str(faces))

# Perform face recognition
face_locations = face_recognition.face_locations(image)
face_loc_list = []
for face in face_locations:
    top, right, bottom, left = face
    face_loc_list.append([left, top, right, bottom])

f = open("../results/faceDetection/hog"+image_name+".txt", "w")
f.write(str(face_loc_list))

def IntersectionOverUnion(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def iou_result(ground_truth, faces):
    iou_of_face = []
    for face in faces:
        iou = []
        for gt in ground_truth:
            iou.append(IntersectionOverUnion(face, gt))
        iou_of_face.append(max(iou))
    avg_iou = sum(iou_of_face)/len(iou_of_face)
    return avg_iou

print("Average IoU using Viola Jones: ", iou_result(ground_truth, faces))
print("Average IoU using HoG: ", iou_result(ground_truth, face_loc_list))