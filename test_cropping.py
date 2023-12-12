import cv2 as cv
import cv2
import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def find_smallest_rectangle(center, points):
    # Unpack center coordinates
    cx, cy = center
    print(cx, cy)

    x_left = 0
    x_right = cx*2
    y_top = 0
    y_bottom = cy*2

    for x, y in points:
        if x > x_left and x < cx:
            x_left = x
        elif x < x_right and x > cx:
            x_right = x
        if y > y_top and y < cy:
            y_top = y
        elif y < y_bottom and y > cy:
            y_bottom = y

    return (x_left, y_top), (x_right, y_top), (x_left, y_bottom), (x_right, y_bottom)

if __name__ == '__main__':
    cap = cv.VideoCapture('./data/lecture.mp4')
    totalframecount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    Vwidth = cap.get(3)  # float `width`
    Vheight = cap.get(4)  # float `height`

    print(Vwidth, Vheight)

    # extract a random frame and hope that there is a slide on it
    # frame_index = random.randint(0, totalframecount)
    frame_index = 2500
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    k = 20
    kernel = np.ones((k, k), np.uint8)
    closing = cv.morphologyEx(frame.copy(), cv.MORPH_CLOSE, kernel)

    # add 1 px black border
    width = 1
    closing[0:width, :] = [0, 0, 0]
    closing[:, 0:width] = [0, 0, 0]
    # bottom
    closing[-width:, :] = [0, 0, 0]
    closing[:, -width:] = [0, 0, 0]


    gray = cv.cvtColor(closing, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.004)
    # dst = cv.dilate(dst, None)
    f = np.ones(frame.shape, dtype=np.uint8) * 255
    f[dst > 0.01 * dst.max()] = [0, 200, 0]

    # get the inidices
    indices = np.where(f == [0, 200, 0])
    X = indices[0]
    Y = indices[1]

    xy = np.array([X, Y]).T
    center_of_image = np.array(frame.shape[:2]) / 2
    left_top, right_top, left_bottom, right_bottom = find_smallest_rectangle(center_of_image, xy)

    print(left_top, right_top, left_bottom, right_bottom)

    # use ffmpeg to crop the image
    width = (right_bottom[0] - left_top[0])//2
    height = (right_bottom[1] - left_top[1])//2
    ffmpeg_cmd = f'ffmpeg -i ./data/lecture.mp4 -filter:v "crop={width}:{height}:{left_top[0]}:{left_top[1]}" -c:a copy ./data/lecture_cropped.mp4 -y'
    print('Running ffmpeg command:', ffmpeg_cmd)
    os.system(ffmpeg_cmd)
