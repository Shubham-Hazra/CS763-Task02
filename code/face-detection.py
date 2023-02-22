import argparse

import cv2 as cv
import face_recognition

cli = argparse.ArgumentParser()
cli.add_argument("--data", help="path to image file", required=True)
cli.add_argument("--faces", help="path to save faces", required=True)
cli.add_argument("--type",type=int, required=True)
args = cli.parse_args()


path_to_save = args.faces
type = args.type

if type == 1:
    image_path = args.data
elif type == 2:
    video_path = args.data

if type == 1:
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    print("Bounding boxes: ")
    print(face_locations)

    for i,face_location in enumerate(face_locations):
        top, right, bottom, left = face_location
        image = cv.imread(image_path)
        face_image = image[top:bottom, left:right]
        image_name = image_path.split("/")[-1]
        image_name = image_name.split(".")[0]
        if i < 10 and len(face_locations) > 1:
             faces = path_to_save +"/" +"face"+image_name + "suffix0" + str(i+1) + ".jpg"
        elif len(face_locations) > 1:
            faces = path_to_save +"/" +"face"+image_name + "suffix" + str(i+1) + ".jpg"
        else:
            faces = path_to_save +"/" +"face"+image_name + ".jpg"
        cv.imwrite(faces, face_image)
        cv.destroyAllWindows()

elif type == 2:
    video = cv.VideoCapture(video_path)
    video_save_path = path_to_save +"/" +"faceVideoOutput"+video_path.split("/")[-1][9:-3] + ".mp4"
    fps = video.get(cv.CAP_PROP_FPS)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    size = (frame_width, frame_height)
    out = cv.VideoWriter(video_save_path, cv.VideoWriter_fourcc(*'mp4v'),fps, size)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        face_locations = face_recognition.face_locations(frame)
        for i,face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            out.write(frame)
    video.release()
    out.release()
        
