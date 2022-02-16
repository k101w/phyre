from __future__ import print_function
import numpy as np
import cv2 as cv
import csv
import pdb
import sys
from random import randint
from utils import Stack
from src.python.phyre.creator.constants import color_to_id
import phyre.interface.shared.ttypes as shared_if
import os

class linedetector:
    def __init__(self):
        self.lines = []
 
    def find_lines(self, frame):
        h, w, ch = frame.shape
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        print(gray.shape)
        print(gray)
        cv.imwrite("gray.png", gray)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
        cv.imwrite("binary.png", binary)
        dist = cv.distanceTransform(binary, cv.DIST_L1, cv.DIST_MASK_3)
        #pdb.set_trace()
        #dist = dist / 15
         #cv.imshow("distance", dist / 15)
        cv.imwrite("dis.png", dist)
        result = np.zeros((h, w), dtype=np.uint8)
        result_set=set()
        ypts = []
        mincol=np.array([0, 351])
        maxcol=np.array([0, 0])
        for row in range(h):
            cx = 0
            cy = 0
            max_d = 0
            for col in range(w):
                d = dist[row][col]
                if d > max_d:
                    max_d = d
                    cx = col
                    cy = row
                    if(col>maxcol[1]): 
                        maxcol[1]=col
                        maxcol[0]=row
                    if(col<mincol[1]): 
                        mincol[1]=col
                        mincol[0]=row
            result[cy][cx] = 255
            result_set.add((cy,cx))
            ypts.append([cx, cy])
        cv.imwrite("skeleton1.png", result)
        xpts = []
        minrow=np.array([365, 0])
        maxrow=np.array([0, 0])
        for col in range(w):
            cx = 0
            cy = 0
            max_d = 0
            for row in range(h):
                d = dist[row][col]
                if d > max_d:
                    max_d = d
                    cx = col
                    cy = row
                    if(row>maxrow[0]): 
                        maxrow[1]=col
                        maxrow[0]=row
                    if(row<minrow[0]): 
                        minrow[1]=col
                        minrow[0]=row
            result[cy][cx] = 255
            result_set.add((cy,cx))
            xpts.append([cx, cy])
 
        cv.imwrite("skeleton2.png", result)

        # cv.line(frame, (mincol[0], mincol[1]), (maxcol[0], maxcol[1]), (0, 0, 255), 2)
        # cv.line(frame, (minrow[0], minrow[1]), (maxrow[0], maxrow[1]), (0, 255, 0), 2)

        # frame = self.line_fitness(ypts, image=frame)
        # frame = self.line_fitness(xpts, image=frame, color=(255, 0, 0))
        rect=cv.minAreaRect(result_set)
        box = cv.boxPoints(rect)
        #box = np.int0(box)
    # 画出边界
        cv.drawContours(frame, [box], 0, (0, 0, 255), 3)
        cv.imwrite("fitlines.png", frame)
        return self.lines
 
    def line_fitness(self, pts, image, color=(0, 0, 255)):
        h, w, ch = image.shape
        [vx, vy, x, y] = cv.fitLine(np.array(pts), cv.DIST_L1, 0, 0.01, 0.01)
        y1 = int((-x * vy / vx) + y)
        y2 = int(((w - x) * vy / vx) + y)
        cv.line(image, (w - 1, y2), (0, y1), color, 2)
        return image
def dra(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)
    le=len(contours)
    if(le==1):
        rect=cv.minAreaRect(contours[0])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print(box)
    
        cv.drawContours(frame, [box], 0, (0, 0, 255), 3)
        cv.imwrite("fitlines.png", frame)

img=cv.imread('new5.jpg')
dra(img)
# a=linedetector()
# a.find_lines(img)
