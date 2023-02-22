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

if type == 1:
    image_path = args.i
elif type == 2:
    video_path = args.i

arnold_encoding = np.array([-0.04622833,  0.13252695,  0.11916612, -0.03984044, -0.05562641,
    0.03548804, -0.0420888 , -0.04895821,  0.05985121,  0.01274674,
    0.25893712,  0.01166112, -0.2990557 , -0.0620512 , -0.09070805,
    0.1237231 , -0.12787671, -0.04951193, -0.12527321, -0.05459099,
    -0.0093086 ,  0.05831843,  0.06773403,  0.02849257, -0.10664611,
    -0.25967833, -0.07464617, -0.06132142, -0.03952987, -0.16263057,
    0.01654664,  0.07112528, -0.16990349, -0.03135378, -0.02244423,
    -0.02238347, -0.08569837, -0.05971587,  0.24385668,  0.10666662,
    -0.1494772 ,  0.13691057, -0.02344902,  0.33729839,  0.30092341,
    -0.00552094, -0.01492577, -0.00964706,  0.13007158, -0.27035528,
    0.03186873,  0.21618612,  0.15176432,  0.03657202,  0.04744587,
    -0.09526722,  0.00884541,  0.16859815, -0.16406977,  0.07234314,
    -0.00587081, -0.18820064,  0.09907995, -0.0397456 ,  0.11495723,
    0.08907981, -0.080827  , -0.12518284,  0.18069458, -0.0866441 ,
    -0.05577801,  0.08550766, -0.1547544 , -0.14239013, -0.28758213,
    0.06529602,  0.34366056,  0.1337298 , -0.21563298,  0.00419354,
    -0.14133796, -0.01518403, -0.00499616,  0.04404867, -0.07992127,
    -0.13180727, -0.11426888, -0.03615835,  0.20549096, -0.01175783,
    -0.10800114,  0.24246469,  0.00392517, -0.03405928,  0.08842244,
    -0.01658491, -0.11404651, -0.02635382, -0.11258332, -0.00268977,
    -0.10242365, -0.19507819, -0.0044045 ,  0.10448996, -0.17435056,
    0.1613165 , -0.00940344,  0.00663572, -0.03972701,  0.01664023,
    -0.09348729,  0.04037151,  0.18441558, -0.27477399,  0.281616  ,
    0.1537327 ,  0.02672058,  0.0816811 ,  0.10452699,  0.08499364,
    -0.07835265,  0.0019085 , -0.05728763, -0.14071079,  0.0537948 ,
    0.00970788,  0.12695609,  0.05034413])
sylvester_encoding = np.array([-0.09098089,  0.0052404 ,  0.01971698, -0.01445911, -0.08169352,
    0.00287552,  0.02134924, -0.11303968,  0.12452722, -0.03728694,
    0.17565402, -0.05421079, -0.24114284,  0.04917074, -0.02386789,
    0.14858468, -0.02648774, -0.05921942, -0.18077306, -0.18163437,
    -0.01401509,  0.00669104, -0.02760838,  0.0437535 , -0.12954757,
    -0.1976051 , -0.0692766 , -0.07231131,  0.03116409, -0.0614039 ,
    -0.02514033, -0.01191539, -0.20780346, -0.10354915,  0.0545804 ,
    0.00389049, -0.01980649, -0.08199795,  0.24679427,  0.00611269,
    -0.10190926,  0.08163711,  0.07646775,  0.26791507,  0.16513339,
    0.04061298,  0.02024611, -0.06554337,  0.14615022, -0.24047743,
    0.04527329,  0.09212283,  0.10437657,  0.0585605 , -0.00916121,
    -0.14829242,  0.03194171,  0.14115277, -0.23169287,  0.14150083,
    0.08619719, -0.11659993, -0.06972851, -0.03471182,  0.26400533,
    0.10919491, -0.14204019, -0.10580339,  0.17365849, -0.12016016,
    0.02438734, -0.05704429, -0.1220646 , -0.15887766, -0.32062846,
    0.13395819,  0.45150185,  0.10599166, -0.21951407,  0.06543404,
    -0.12316862,  0.03026369,  0.03467617,  0.03492052, -0.08292989,
    -0.04853889, -0.03302751,  0.01932486,  0.23411551, -0.02421142,
    -0.01681693,  0.25271845,  0.01221341, -0.07625362,  0.04388457,
    0.12238982, -0.17694119, -0.1039461 , -0.06471294, -0.00988032,
    0.08683933, -0.15608677,  0.14317526,  0.13782182, -0.27395561,
    0.20541273, -0.07705434, -0.0130765 ,  0.02851853,  0.0628564 ,
    -0.08255742, -0.05369024,  0.16677681, -0.2335344 ,  0.18812877,
    0.15379953,  0.02029939,  0.13264604,  0.05337957,  0.12034331,
    0.01168956,  0.15244527, -0.10749183, -0.12629099,  0.01138368,
    0.01228185,  0.0898255 ,  0.14061931])
known_face_encodings = [arnold_encoding, sylvester_encoding]
recognition_dict = {0: "Arnold Schwarzenegger", 1: "Sylvester Stallone"}

if type == 1:
    image = cv.imread(image_path)
    unknown_image = face_recognition.load_image_file(image_path)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image)
    face_locations = face_recognition.face_locations(unknown_image)
    face_names = []
    for face in unknown_face_encodings:
        results = face_recognition.compare_faces(known_face_encodings, face, tolerance=0.5)
        name = None
        if results[0]:
            name = recognition_dict[0]
        elif results[1]:
            name = recognition_dict[1]
        else:
            name = "Unknown"
        face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue
            cv.rectangle(image, (left-20, top-20), (right+20, bottom+20), (0, 0, 255), 3)
            cv.rectangle(image, (left-20, bottom+10), (right+20, bottom+20), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(image, name, (left-20, bottom+6), font, 0.75, (255, 0, 0), 1)
    cv.imwrite(path_to_save, image)


elif type == 2:
    video_capture = cv.VideoCapture(video_path)
    fps = video_capture.get(cv.CAP_PROP_FPS)
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    size = (frame_width, frame_height)
    out = cv.VideoWriter(path_to_save, cv.VideoWriter_fourcc(*'mp4v'), fps, size)
    face_locations = []
    face_encodings = []
    face_names = []
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_names = []
        for face in face_encodings:
            results = face_recognition.compare_faces(known_face_encodings, face)
            name = None
            if results[0]:
                name = recognition_dict[0]
            elif results[1]:
                name = recognition_dict[1]
            else:
                name = "Unknown"
            face_names.append(name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue
            cv.rectangle(frame, (left-20, top-20), (right+20, bottom+20), (0, 0, 255), 3)
            cv.rectangle(frame, (left-20, bottom+10), (right+20, bottom+20), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left-20, bottom+6), font, 1, (0, 255, 0), 2)     
        out.write(frame)
    video_capture.release()
    cv.destroyAllWindows()
    out.release()