import argparse
import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

fps_time = 0

#######START#######
showBG = True
model = 'mobilenet_thin'
vidPath = 'images/video_Trim.mp4'
print('initialization %s : %s' % (model, get_graph_path(model)))
w, h = 432, 368
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
# cap = cv2.VideoCapture(vidPath)
cap = cv2.VideoCapture(0)
resize_out_ratio = 4.0

print("displaying...")
if cap.isOpened() is False:
    print("Error opening video stream or file")
while cap.isOpened():
    ret_val, image = cap.read()
    image = cv2.resize(image, (w, h))
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
    if not showBG:  # show skeleton only
        image = np.zeros(image.shape)
    image, rightAngle, leftAngle = TfPoseEstimator.draw_arm(image, humans, imgcopy=False)
    print("right : ", rightAngle, "   left : ", leftAngle)

    ###pose detect working/gaming
    if rightAngle == 0 and leftAngle == 0:
        cv2.putText(image, "arms not detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif 0 < rightAngle < 60 or 0 < leftAngle < 60:
        cv2.putText(image, "playing games", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    elif 60 <= rightAngle <= 120 or 60 <= leftAngle <= 120:
        cv2.putText(image, "working", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    cv2.imshow('tf-pose-estimation result', image)
    fps_time = time.time()
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
print('finished...')
