# Import required packages
import argparse
import pickle
import sys

import cv2 as cv


# Function to correct the bounding box coordinates
def correct_bounding_box(list):
    if list[0] > list[2]:
        temp = list[0]
        list[0] = list[2]
        list[2] = temp
    if list[1] > list[3]:
        temp = list[1]
        list[1] = list[3]
        list[3] = temp
    return list

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='path to folder containing images/path to video', required=True)
parser.add_argument('--type', type=int, required=True)
args = parser.parse_args()

video_path = None
folder_path = None

# Check if the data is image or video
image_path = args.data
image_name = image_path.split('/')[-1]
annotation_path = "../data/captured/"+image_name.split('.')[0] + '.txt'

# If data is image, get the list of image paths
image_list = [image_path]

# If type is 1, get the bounding box coordinates
if args.type == 1:
    image_data = []
    upper_left = (-1, -1)
    lower_right = (-1, -1)
    left_coord = True
    drawing = False
    fix_rect = False

    # Function to get the bounding box coordinates using mouse
    def get_coordinates(event, x, y, flags, param):
        global upper_left, lower_right, left_coord, drawing, image_data, fix_rect
        if event == cv.EVENT_LBUTTONDOWN:
            fix_rect = False
            drawing = True
            upper_left = (x, y)
            image_data.append([x, y])
        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                lower_right = (x, y)
        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            image_data[-1] += [x, y]
            if upper_left != (-1, -1):
                fix_rect = True

    # Function to get the bounding box coordinates from the image
    def get_data(img):
        global upper_left, lower_right, fix_rect
        cv.namedWindow('image')
        img1 = img.copy()
        cv.setMouseCallback('image', get_coordinates)
        while True:
            if fix_rect:
                cv.rectangle(img1, upper_left, lower_right, (255, 0, 0), 2)
                upper_left = (-1, -1)
                lower_right = (-1, -1)
            img_copy = img1.copy()
            if upper_left != (-1, -1) and lower_right != (-1, -1):
                cv.rectangle(img_copy, upper_left, lower_right, (255, 0, 0), 2)
            cv.imshow('image', img_copy)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cv.destroyAllWindows()
    bounding_box = []
    for image_path in image_list:
        image  = cv.imread(image_path)
        get_data(image)
        bounding_box.append(image_data)
        image_data = []
    f = open(annotation_path, 'wb')
    print(bounding_box[0])
    f.write(pickle.dumps(bounding_box[0]))
    f.close()
    sys.exit(0)