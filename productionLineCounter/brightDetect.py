import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
from datetime import datetime, timedelta


def read_brightness(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([imgGray], [0], None, [256], [0, 256])
    pixels = sum(hist)
    brightness = 0
    for idx in range(0, 256):
        ratio = hist[idx] / pixels
        brightness += ratio * idx
    text = str(brightness)
    cv2.putText(img,text,(600,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
    return brightness


bright = cv2.imread("demo_bright.jpg")
# dark = cv2.imread("demo_dark.jpg")
print(read_brightness(bright))
# print(read_brightness(dark))
cv2.imshow("b",bright)
cv2.waitKey(0)
#

d = 1
result = {"Prediction_break": []}

path = r"D:\SuperAILevel2\week4\Denso\test_video"
day_folder = [f + "\\" + "CtlEquip_10" for f in os.listdir(path) if "." not in f]
video_list = [v for v in os.listdir(os.path.join(path, day_folder[d])) if ".csv" not in v]

for i in range(5,6):
    data = {"timestamp": [], "frame": [], "file": [], "brightness": [], "is_breaktime": []}
    # Display video
    video_name = video_list[i]
    video_path = path + "\\" + day_folder[d] + "\\" + video_name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 15
    count = 0
    try:
        year = int(video_name[-18: -14])
        month = int(video_name[-14: -12])
        day = int(video_name[-12: -10])
        hr = int(video_name[-10: -8])
        m = int(video_name[-8: -6])
        sec = int(video_name[-6: -4])

        start_time = datetime(year, month, day, hr, m, sec)
        print(start_time.strftime("%c"))
    except:
        pass

    while cap.isOpened():
        ret, frame = cap.read()

        if ret and int(count % (fps * 30)) == 0:
            bn = read_brightness(frame)
            cv2.imshow("frame", frame)

            t = timedelta(seconds=(count / fps))
            current_time = start_time + t

            if bn < 100:
                bt = 1
            else:
                bt = 0

            data["file"].append(video_name)
            data["frame"].append(count)
            data["brightness"].append(bn)
            data["is_breaktime"].append(bt)
            data["timestamp"].append(current_time)
            # print("Brightness of frame %s: " % (count), bn)
            # print("Breaktime: ", bt)
            # print("Current time: ", current_time)
            # print("time stamp current frame: ", count / fps)
            # print("-" * 10)


        if cv2.waitKey(1) == ord("q") or not ret:
            break
        # print(count, ret,int(count % fps*20))
        count += 1

    cap.release()
    cv2.destroyAllWindows()

    data = pd.DataFrame(data)
    data = data.drop(data.index[0:3])
    print(data)

    if np.sum(data["is_breaktime"]) > 3:
        result["Prediction_break"].append("Yes")
    else:
        result["Prediction_break"].append("No")

result = pd.DataFrame(result)
result.to_csv("D:\SuperAILevel2\week4\Denso" +"\\" + "result31" + ".csv", index=1)
