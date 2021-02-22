import numpy as np
import os
import cv2
from math import radians, sin, cos, tan
from statistics import mode
import time
import matplotlib.pyplot as plt
import copy


def rot_z(theta):
    r = radians(theta)
    c = cos(r)
    s = sin(r)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def rot_y(theta):
    r = radians(theta)
    c = cos(r)
    s = sin(r)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def rot_x(theta):
    r = radians(theta)
    c = cos(r)
    s = sin(r)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def skyDetect(image):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for x in range(0, 512, 1):
        col_pix = []
        col_pix.append(imageGray[0][x])
        is_skyline = False
        for y in range(1, 512):

            pixel = imageGray[y][x]  # from top to bottom
            # 255 = white, 0 = black
            col_pix.append(pixel)
            # print(pixel)

            # grey jump
            if np.abs(col_pix[y] - col_pix[y - 1]) > 30 and col_pix[y] < 100:
                # jump buffer
                for count in range(1, 6):  # 1-5
                    if col_pix[y] > 100:  # found white
                        is_skyline = False
                        break
                    y += 1
                    pixel = imageGray[y][x]
                    col_pix.append(pixel)
                    is_skyline = True
                    print("y", y)
                    print("count", count)

                if is_skyline:
                    # image[y][x] = (0, 255, 0)
                    cv2.circle(image, (x, y - 5), 5, (0, 255, 0), 1)
                    break

        # print(col_pix)

    return image


def rot_y(a):
    r = radians(a)
    c = cos(r)
    s = sin(r)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def sky3D(ds_id, sider, frame, r_x, r_y, r_z):
    class CameraExtrinsic(object):
        def __init__(self, camera_extrinsic_path="D:\SuperAILevel2\week5\HighwayDOH\set1\camera.csv"):
            lines = np.loadtxt(camera_extrinsic_path, delimiter=',', dtype=np.str)
            self.cam_extrinsic_by_hash = {
                self.cam_hash(*line[:2]): (np.float32(line[2:11]).reshape((3, 3)), np.float32(line[11:])) for line in
                lines}

        def cam_hash(self, ds_id, frame_no):
            return f"{ds_id}_{frame_no}"

        def cam_extrinsic(self, ds_id, frame_no):
            return self.cam_extrinsic_by_hash[self.cam_hash(ds_id, frame_no)]

    cams = CameraExtrinsic()
    ds_path = os.path.join("/ds\\", ds_id)
    frame_no = frame
    fov, near, far = 90, 1, 1100
    fd = 1 / tan(radians(fov) / 2)
    dfn = far - near
    proj_mat = np.array([
        [fd, 0, 0, 0],
        [0, fd, 0, 0],
        [0, 0, -(far + near) / dfn, -1],
        [0, 0, -2 * far * near / dfn, 0]])

    rotate_mat, cam_pos = cams.cam_extrinsic(ds_id, frame_no)
    pcs = np.load(os.path.join(ds_path, "pc.npz"))["data"]
    # world coordinate to camera coordinate
    pcs = pcs - cam_pos
    is_near = (abs(pcs[:, 0]) + abs(pcs[:, 1]) < 50)
    pcs = pcs[is_near]

    rotate_mat_prime = np.eye(3)
    rotate_mat_prime = rot_x(r_x) @ rot_y(r_y) @ rot_z(r_z)

    switcher = {
        'F': 0,
        'L': -90,
        'R': 90
    }

    side = sider
    deg = switcher.get(side)
    # rotate pcs
    inv_rotate_mat = (rot_y(deg) @ rotate_mat @ rotate_mat_prime).T
    locals_ = pcs @ inv_rotate_mat
    # simple way to remove invisible points (or using frustum culling)
    front = locals_[locals_[:, 2] <= -1]
    img_ori = cv2.imread(os.path.join(ds_path, f"{frame_no:02d}_{side}_img.png"))
    img = copy.deepcopy(img_ori)
    blank = np.zeros((512, 512, 1))
    if img is None:
        print("image not found")
        img = np.zeros((512, 512, 3))

    # from camera coordinate to normalized device coordinate
    ndc = np.c_[front, np.ones(front.shape[0])] @ proj_mat
    w = ndc[:, 3] / 256

    # from ndc to screen coordinate 512x512
    x = np.array(ndc[:, 0] / w + 256, dtype=np.int)
    y = np.array(-ndc[:, 1] / w + 256, dtype=np.int)

    cond = np.where((x >= 0) & (x < 512) & (y >= 0) & (y < 512))
    scx = y[cond], x[cond]
    img[scx] = (0, 0, 255)
    cv2.imwrite(f"{side}.png", img)
    cv2.imshow("img", img)
    blank[scx] = 1
    cv2.imshow("b", blank)

    kernel = np.ones((1, 1), np.uint8)
    erode = cv2.erode(blank, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    cv2.imshow("res", dilate)

    pixels = []
    h = []  # col coordinate
    v1 = []  # row coordinate 3d
    v2 = []  # row coordinate 2d
    im = cv2.imread(r"/ds\0b61415ee77f41a39c9c4140da2476c2\00_L_img.png")

    ### find skyline
    points = []
    for x in range(512):  # row
        for y in range(512):  # col
            if dilate[y][x] == 1:
                points.append((x, y))
                h.append(x)
                v1.append(y)

                break
    temp = copy.deepcopy(v1)
    temp.sort()

    h_new = []
    v1_new = []
    for i in range(0, 200, 20):
        v1_new.append(temp[i])
        h_new.append(h[v1.index(temp[i])])
        cv2.circle(im, (h[v1.index(temp[i])], temp[i]), 5, (0, 0, 255), 1)

    cv2.imshow("im", im)
    cv2.waitKey(0)

    return h_new, v1_new, front

h,v1,_ = sky3D("0b61415ee77f41a39c9c4140da2476c2", 'L',00,0,0,0)

print(h)
print(mode(v1))
print(v1)
