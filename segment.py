#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
import argparse
import math
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())


def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)  # 消失的像素是skeleton的一部分
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break

    return skel


def skeleton_endpoints(skel):
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates,
    # you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered == 11)] = 1
    return out


img = cv2.imread(args["image"], 0)
# img = cv2.bitwise_not(img,img)
# gray = cv2.imread(args["image"],0)
gray = img
height, width = gray.shape

kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
erode = cv2.erode(gray, kernel1, iterations=1)
cv2.imshow('erode', erode)
cv2.waitKey(0)

ret, img = cv2.threshold(img, 127, 255, 0)

skel = img  # skeletonize(img)

cv2.imshow("skel", skel)
cv2.waitKey(0)

# out = skeleton_endpoints(skel)
# cv2.imshow("out", out)
# cv2.waitKey(0)
# sys.exit()

res = np.zeros((height, width))
im = skel
for x in range(1, height - 1):
    for y in range(1, width - 1):
        if im[x, y] == 0:
            continue
        neighbors = 0
        for k in range(-1, 1 + 1):
            for l in range(-1, 1 + 1):
                if not k and not l:
                    continue
                neighbors += im[x+k, y+l]
        if neighbors/255 > 2:
            res[x, y] = 255
            # im[x, y] = 0

for x in range(1, height - 1):
    for y in range(1, width - 1):
        if res[x, y] == 0:
            continue
        neighbors = (res[x+1, y] + res[x-1, y] + res[x, y+1] + res[x, y-1])/255
        res[x, y] = 255 if neighbors > 1 or neighbors == 0 else 0

cv2.imshow('branchpoints', res)
cv2.imwrite('data/tmp.jpg', res)
cv2.waitKey(0)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 2*height+1))
close = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel2)
cv2.imshow('close', close)
cv2.waitKey(0)

element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
close = cv2.erode(close, element)
cv2.imshow('segmentline', close)
cv2.waitKey(0)

cv2.destroyAllWindows()
