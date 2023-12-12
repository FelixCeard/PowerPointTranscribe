import cv2 as cv
import cv2
import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


# project
from modules import cropping as crp



if __name__ == '__main__':

    cap = cv.VideoCapture('./data/lecture.mp4')
    totalframecount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # crop manually
    # crp.manual(cap)
    crp.crop(cap, 'manual')