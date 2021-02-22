import cv2
from cv2 import VideoWriter
import os, sys
import numpy as np

frameCount = 0
path_out = r"D:\SuperAILevel2\week4\test_cut_video224weeee"
path_src = r"D:\SuperAILevel2\week4\test_video"

d=1 # change vide date
day_folder = [f + "\\" + "CtlEquip_10" for f in os.listdir(path_src) if "." not in f]
video_list = [v for v in os.listdir(os.path.join(path_src, day_folder[d])) if ".csv" not in v]

for i in range(len(video_list)):

    video_name = video_list[i]
    video_path = path_src + "\\" + day_folder[d] + "\\" + video_name
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # 15
    start_frame_count = frame_count - fps * 2
    stop_frame_count = frame_count+50

    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - fps * 5)  # *cap.get(cv2.CAP_PROP_FPS)

    print(frame_count)
    print(start_frame_count)
    print(stop_frame_count)

    out = None
    frameCount = start_frame_count

    while cap.isOpened():
        ret, frame = cap.read()
        frameCount += 1

        if cv2.waitKey(1) == ord("q") or not ret:
            break
        cv2.imshow("f", frame)
        if not out:
            height, width, channels = frame.shape
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            filename = path_out + "\\" + str(i) + ".avi"
            out = cv2.VideoWriter(filename=filename, fourcc=fourcc, fps=fps, frameSize=(width, height))
        if start_frame_count <= frameCount < stop_frame_count:
            out.write(frame)
            print("writing")
        elif frameCount == stop_frame_count:
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print("finish...")
