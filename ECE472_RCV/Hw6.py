# Yiming Bi 146004795 yb127
# Homework 6

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math

# this function will find result points of four corners of pic1
def find_forward_warping_points(width,height,a,b,c,d,e,f):
    pt = []
    p0 = np.array([a*0+b*0+c, d*0+e*0+f])
    p1 = np.array([a*0+b*height+c, d*0+e*height+f+height])
    pt = np.vstack([p0,p1])
    p2 = np.array([a*width+b*0+c+width, d*width+e*0+f])
    pt = np.vstack([pt,p2])
    p3 = np.array([a*width+b*height+c+width, d*width+e*height+f+height])
    pt = np.vstack([pt,p3])
    return pt

# this function do bilinear interpolation
def find_intensity_weight(x,y):
    x_ceil = math.ceil(x)
    x_floor = math.floor(x)
    y_ceil = math.ceil(y)
    y_floor = math.floor(y)
    left = x - x_floor
    right = x_ceil - x
    up = y - y_floor
    down = y_ceil - y
    a = right*down
    b = left*down
    c = right*up
    d = left*up
    weight = np.array([a,b,c,d])
    return weight

def p1():
    pic_old = cv.imread('C:\\pic\\01.jpg')
    height, width, depth = pic_old.shape
    pt_new = find_forward_warping_points(width,height,0.02,0.01,10,0.01,-0.02,5)
    pic_new = np.empty([int(math.ceil(pt_new[3,1])),int(math.ceil(pt_new[3,0])),3])
    pt_old = np.array([[0,0],
                       [0,height],
                       [width,0],
                       [width,height]])
    F, uesless_value = cv.findHomography(pt_new, pt_old)

    for i in range(int(math.ceil(pt_new[3,1]))):
        for j in range(int(math.ceil(pt_new[3,0]))):
            pt = np.matmul(F,np.array([[i],[j],[1]]))
            x = float(pt[0])
            y = float(pt[1])
            if y > width-1 or x > height-1:
                continue
            w = find_intensity_weight(x,y)
            new_int = pic_old[math.floor(x), math.floor(y), :] * w[0] + \
                      pic_old[math.ceil(x), math.floor(y), :] * w[1] + \
                      pic_old[math.floor(x), math.ceil(y), :] * w[2] + \
                      pic_old[math.ceil(x), math.ceil(y), :] * w[3]
            pic_new[i,j,:] = new_int
    pic_new = cv.convertScaleAbs(pic_new)
    pic_new_new = pic_old[:597,:381,:]*0.5 + pic_new[:597,:381,:]*0.5

    cv.imshow('pic_new', cv.convertScaleAbs(pic_new_new))
    cv.waitKey(0)
    cv.destroyAllWindows()

p1()