import argparse
import logging
import sys
import time

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

########START##########


resize_out_ratio = 4.0
w = 432
h = 368
imPath = 'images/act2.jpg'
e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w,h))

# estimate human poses from a single image !
image = common.read_imgfile(imPath, None, None)
if image is None:
    print('Image can not be read, path=%s' % imPath)
    sys.exit(-1)

t = time.time()
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio)
elapsed = time.time() - t

print(humans)
print('inference image: %s in %.4f seconds.' % (imPath, elapsed))
for human in humans:
    print(human,"\n")

image,rightAngle,leftAngle = TfPoseEstimator.draw_arm(image, humans, imgcopy=False)
print("right : ",rightAngle)
print("left : ",leftAngle)
try:
    import matplotlib.pyplot as plt

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
    bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)

    plt.show()
except Exception as e:
    print('matplitlib error, %s' % e)
    cv2.imshow('result', image)
    cv2.waitKey()
