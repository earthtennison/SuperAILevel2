import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import re


def detect_obj(path):
    cap = cv2.VideoCapture(path)

    topX = 630
    topY = 78
    botomX = 564
    botomY = 60

    # image = cv2.line(image, start_point, end_point, color, thickness)
    bg_sub = cv2.createBackgroundSubtractorKNN()
    count = 0
    detect = 0
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # L1 =[525:595]
        # L2 = np.r_[100:150]

        if not ret:
            break

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame[botomY:topY, botomX:topX, :], cv2.COLOR_BGR2GRAY)
        sensor = bg_sub.apply(gray)
        # image = cv2.rectangle(frame, (508, 29), (666, 306), (255, 0, 0), 5)
        # print(np.sum(sensor))
        if np.sum(sensor) > 50000 and count > 10:
            frame[botomY:topY, botomX:topX, :2] = 0
            detect += 1
        else:
            frame[botomY:topY, botomX:topX, 1:] = 0

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow("sensor", sensor)
        # cv2.imshow("g",gray)

        if cv2.waitKey(10) == ord("q") or not ret:
            break

        count += 1

    return "Yes" if detect > 5 else "No"


path = r"D:\SuperAILevel2\"
video_list = [v for v in os.listdir(path) if ".avi" in v]
video_list.sort(key=lambda f: int(re.sub('\D', '', f)))
# print(video_list)
result = {"Prediction_object": []}
for i in range(len(video_list)):
    full_path = path + "\\" + video_list[i]
    is_object = detect_obj(full_path)
    print(str(i)+".avi", is_object)
    print("-" * 10)
    result["Prediction_object"].append(is_object)

# detect_obj(path + "\\" + video_list[17]) #test short object

result = pd.DataFrame(result)
result.to_csv("D:\SuperAILevel2\week4" + "\\" + "result_detect30" + ".csv", index=1)
