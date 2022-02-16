from __future__ import print_function
from errno import EILSEQ
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
import math
from copy import deepcopy
import imageio
import math
#双线性插值 超分辨率
#

def change_float(line):
    new=np.array_like(line)
    for i in range(line.shape[0]):
        for j in range(line.shape[1]):
            new[i][j]=float(line[i][j])
    return new

def capture(img):
    #img=cv.GaussianBlur(img, (3, 3), 0)
    kernel=np.ones((5, 5), np.uint8)
    cv.imwrite('ini0.png',img)
    imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(imgray,200,255,cv.THRESH_BINARY_INV)
    #cv.imwrite('ini1.png',thresh)
    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
    # cv.imwrite('ini1.png',opening)
    edges = cv.Canny(thresh, 50, 100,apertureSize=5)
    #cv.imwrite('ini2.png',edges)
    #TODO: draw stick and jar
    # bar=[x for x in line if x[5]==1]
    # ball=np.array([x for x in line if x[4]==1.0])
    # jar=[x for x in line if x[6]==1]
    # stick=[x for x in line if x[7]==1]
    # colors_cir=rand_color(ball.shape[0])
    # #colors_bar=rand_color(bar.shape[0])
    # contours=[]
    # empty=cv.imread('empty.png')
    # for i in range(len(ball)):
    #     cv.circle(img, (ball[i][0],ball[i][1]), int(ball[i][3]/2), colors_cir[i], 2)
    # cv.imwrite('ini0_1.png',img)
    # pdb.set_trace()
    # for i in len(bar):
    #     cv.rectangle(img,)

    #TODO: the initial image is not smooth enough
    contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)
    print(len(contours))
    colors=rand_color(len(contours))
    # Circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=30,minRadius=0,maxRadius=0)
    # pdb.set_trace()
    # Circles = Circles.reshape(-1,Circles.shape[-1])
    # colors_cir=rand_color(Circles.shape[0])
    # for i in range(Circles.shape[0]):
    #     print(Circles[i])
    #     cv.circle(img, (Circles[i][0],Circles[i][1]), int(Circles[i][2]), colors_cir[i], 2)
    # cv.imwrite('ty.png',img)
    # pdb.set_trace()
    boxes=[]

    for i in range(len(contours)):
        if(i==0):
            im1=cv.drawContours(img,contours[i],-1,colors[i],1)
        else:
            im1=cv.drawContours(im1,contours[i],-1,colors[i],1)
        boxes.append(cv.boundingRect(contours[i]))
    
    im1=drawRectangles(boxes,im1,colors)
    return im1,boxes,colors
#边缘咋办


def getDist_P2L(PointP,Pointa,Pointb):
    """计算点到直线的距离
        PointP：定点坐标
        Pointa：直线a点坐标
        Pointb：直线b点坐标
    """
    #求直线方程
    A=0
    B=0
    C=0
    A=Pointa[1]-Pointb[1]
    B=Pointb[0]-Pointa[0]
    C=Pointa[0]*Pointb[1]-Pointa[1]*Pointb[0]
    #代入点到直线距离公式
    distance=0
    distance=(A*PointP[0]+B*PointP[1]+C)/math.sqrt(A*A+B*B)
    
    return distance

def dra(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)
    le=len(contours)
    if(le==1):
        rect=cv.minAreaRect(contours[0])
        box = cv.boxPoints(rect)
        box = np.int0(box)#box[0] box[3]对角线
        
    
    cv.drawContours(frame, [box], 0, (0, 0, 255), 3)
    cv.imwrite("fitlines.png", frame)


def find_box(mask):
  obj=(mask.astype(int)*255).astype(np.uint8)
  cv.imwrite('np.jpg',obj)
  contours, hierarchy = cv.findContours(obj,cv.RETR_TREE,cv.CHAIN_APPROX_TC89_KCOS)

  #pdb.set_trace()
  rect=cv.minAreaRect(contours[0])
  box = cv.boxPoints(rect)
  box = np.int0(box)
  return box



def capture_new(img):
  img1=img.reshape(-1,3)
  a,b=np.unique(img1,axis=0,return_counts=True)
  imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
  imghsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
  ret,thresh = cv.threshold(imgray,200,255,cv.THRESH_BINARY_INV)
  num_labels, labels,labels_color,label_area = connectedComponents_color(img)
  #output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
  boxes=[]
  #output[:,:,:]=254
  for i in range(0, num_labels):
    mask = labels == i
    # output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    # output[:,:,:]=254
    # if(obj.sum()==1): 
    #   labels[mask]=0
    #   num_labels-=1
    #   labels_color.pop(i-1)
    #   continue
    # label_area.append(obj.sum())
    # record the minmun rectangles
    box=find_box(mask)
    boxes.append(box)
    #TODO:
    #if(labels_color[i-1]==5): dra(output)

  print(num_labels)
  print(labels_color)
  colors=rand_color(num_labels,labels_color)
  im1=drawRectangles(boxes,img,colors)
  #cv.imwrite('cap_new.jpg',im1)
  return im1,num_labels, labels,labels_color,label_area,colors,boxes
  #pdb.set_trace()

#RGB
#白色是254，254，254
# White.
# Red.(243,79,70)
# Green.(107,206,187)
# Blue.(24,119,242)
# Purple.(75,74,164)
# Gray.(185,202,210)
# Black.(0,0,0)
def Judge(pixel):
  if(0<=pixel[0]<5 and 0<=pixel[1]<5 and 0<=pixel[2]<5): return 1#color_to_id('black')
  if(20<pixel[0]<26 and 115<pixel[1]<120 and 238<pixel[2]<245): return 2 #color_to_id('blue')
  if(70<pixel[0]<80 and 70<pixel[1]<80 and 160<pixel[2]<170): return 3 #color_to_id('purple')
  if(103<pixel[0]<200 and 203<pixel[1]<210 and 184<pixel[2]<190): return 4 #color_to_id('green')
  if(180<pixel[0]<190 and 198<pixel[1]<204 and 206<pixel[2]<214): return 5 #color_to_id('gray')
  if(240<pixel[0]<246 and 66<pixel[1]<83 and 67<pixel[2]<73): return 6 #color_to_id('red')
  else:return 0


def rand_color(num,labels_color):
    colors=[]
    for i in range(num):
        colors.append((np.random.uniform(0,255), np.random.uniform(0, 255), np.random.uniform(0, 255)))
        # if(labels_color[i]==1):
        #   colors.append((0,0,0))
        # if(labels_color[i]==2):
        #   colors.append((24,119,242))
        # if(labels_color[i]==3):
        #   colors.append((75,74,164))
        # if(labels_color[i]==4):
        #   colors.append((107,206,187))
        # if(labels_color[i]==5):
        #   colors.append((185,202,210))
        # if(labels_color[i]==6):
        #   colors.append((243,79,70))

    return colors


def connectedComponents_color(image):
  num_labels=0
  image=cv.cvtColor(image,cv.COLOR_BGR2RGB)
  #pdb.set_trace()
  labels=np.zeros((image.shape[0],image.shape[1]))
  labels[:,:]=-1
  #pdb.set_trace()
  label_color=[]
  label_area=[]
  k=-1
  basket=Stack()
  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      count=0
      if ((image[i][j]!=[254,254,254]).any() and labels[i][j]==-1):
        co=Judge(image[i][j])
        #print(co)
        if(co==0): continue
        k+=1
        num_labels+=1
        basket.push((i,j))
        #pdb.set_trace()
        while(basket.empty()==False):
          x,y=basket.pop()
          #pdb.set_trace()
          labels[x][y]=k
          count+=1
          #print(count)
          if(y+1<image.shape[1] and (image[x][y+1]==image[x][y]).all() and labels[x][y+1]==-1):
             basket.push((x,y+1))
          if(x+1<image.shape[0] and (image[x+1][y]==image[x][y]).all() and labels[x+1][y]==-1):
             basket.push((x+1,y))
          if(y-1>=0 and (image[x][y-1]==image[x][y]).all() and labels[x][y-1]==-1):
             basket.push((x,y-1))
          if(x-1>=0 and (image[x-1][y]==image[x][y]).all() and labels[x-1][y]==-1):
             basket.push((x-1,y))
          if(x-1>=0 and y-1>=0 and (image[x-1][y-1]==image[x][y]).all() and labels[x-1][y-1]==-1):
             basket.push((x-1,y-1))
          if(x+1<image.shape[0] and y-1>=0 and (image[x+1][y-1]==image[x][y]).all() and labels[x+1][y-1]==-1):
             basket.push((x+1,y-1))
          if(x-1>=0 and y+1<image.shape[1] and (image[x-1][y+1]==image[x][y]).all() and labels[x-1][y+1]==-1):
             basket.push((x-1,y+1))
          if(x+1<image.shape[0] and y+1<image.shape[1] and (image[x+1][y+1]==image[x][y]).all() and labels[x+1][y+1]==-1):
             basket.push((x+1,y+1))
        mask=labels==k
        #pdb.set_trace()
        obj=(mask.astype(int)).astype(np.uint8)
        if(obj.sum()==1): #too small pixels are not considered
          labels[mask]=-1
          num_labels-=1
          k-=1
        else:
          label_color.append(co)
          label_area.append(obj.sum())
          #print(k)
  return num_labels,labels,label_color,label_area



def drawRectangles(boxes,pic,colors):
    for i, box in enumerate(boxes):
        # p1 = (box[0][0], box[0][1])
        # p2 = (box[2][0], box[2][1])
        # #pdb.set_trace()
        # cv.rectangle(pic, p1, p2, colors[i], 2, 1)
        cv.drawContours(pic, [box], 0, colors[i], 2)
    return pic

def putnum(boxes,check,frame):
  for i , box in enumerate(boxes):
    p1=int((box[0][0]+box[2][0])/2)
    p2=int((box[0][1]+box[2][1])/2)
    try:
      l=len(check[i])
      for j in range(l):
        p1+=10
        cv.putText(frame,'{}'.format(check[i][j]),(p1,p2),cv.FONT_HERSHEY_COMPLEX, 0.4, (164,74,75), 1)
    except(TypeError):
      cv.putText(frame,'{}'.format(check[i]),(p1,p2),cv.FONT_HERSHEY_COMPLEX, 0.4, (164,74,75), 1)
  return frame


class my_create():
  
  def __init__(self):
    self.mask = []
  
  def create(self,num_labels, labels,labels_color,label_area,boxes,colors):
    #ini: 第一帧里面的信息
    #last:上一帧里面的信息
    self.ini_labels=labels
    self.num_labels=num_labels
    self.ini_labels_color=np.array(labels_color)
    self.ini_label_area=np.array(label_area).astype(np.int32)
    self.ini_boxes=np.array(boxes)
#last 可以指未发生碰撞前的最新状态
    self.last_labels=self.ini_labels #保留上一帧的mask
    self.last_labels_color=self.ini_labels_color
    self.last_boxes=self.ini_boxes  #保留上一帧的box
    self.last_label_area=self.ini_label_area  #保留上一帧的每个图片的面积
    self.check=[] #保留上一帧的图像的label顺序

    self.colors=colors

    # 碰撞相关
    self.collade=False
    self.col_index=[]
    self.col_mask=[]
    self.col_area=np.empty(0)
    self.col_color=np.empty(0)
    self.col_boxes=[]
    
    
  
  def fitness(self,index,index_collade,box,flag):# deal with multiple index situation L1:
    dis=[(abs(self.last_boxes[i]-box)).sum() for i in index]
    if(self.collade==True):
      dis_col=[(abs(np.array(self.col_boxes)[i]-box)).sum() for i in index_collade]
      if(len(dis)==0):
        flag=True
        best_index=[index_collade[np.argmin(dis_col)]]
      elif(len(dis_col)==0):
        flag=False
        best_index=[index[np.argmin(dis)]]
      else:
        if(np.min(dis)<=np.min(dis_col)):
          flag=False
          best_index=[index[np.argmin(dis)]]
        else:
          flag=True
          best_index=[index_collade[np.argmin(dis_col)]]
    else:
      flag=False
      if(len(dis)==0): pdb.set_trace()
      best_index=[index[np.argmin(dis)]]
    #pdb.set_trace()
    return best_index,flag

  def collision(self,label_area,i,color,mask):#必须有collision labels
    overlap=np.empty((0))
    lap_index=[]
    #overlap={}
    if(self.collade==True):#曾经有过碰撞
      for j in range(len(self.col_mask)):
        new_mask=np.multiply(self.col_mask[j],mask)
        #重叠部分是否就是原来的碰撞部分
        lap=np.sum(new_mask.astype(int))/np.sum(self.col_mask[j].astype(int))
        overlap=np.append(overlap,lap)
        lap_index.append(self.col_index[j])
    # if(i==7 and label_area[i]==1290): 
    #   print('ATTENTION')
    #   pdb.set_trace()
    for j in range(self.num_labels):
      if(self.last_labels_color[j]==color):
          if(self.collade==True and np.array([self.col_index[k].count(j) for k in range(len(self.col_index))]).sum()!=0): continue #已经被碰撞检测过了
          new_mask=np.multiply(self.last_labels==j,mask)
          lap=np.sum(new_mask.astype(int))/self.last_label_area[j] #重叠面积占比
          #overlap.update({lap:j})
          overlap=np.append(overlap,lap)
          lap_index.append(j)
    area=0
    indexs=[]
    print('collison overlap')
    print(overlap)
    #pdb.set_trace()
    #lap=np.array([])
    while(abs(area-label_area[i])>50):#找到有多少object连在一起
      max=np.max(overlap)
      if(max==0):#重叠部分没有了，可能是运动的太快#也有可能是畸变太严重
        if(area==0):
          print('waiting to be fixed')
          pdb.set_trace()
        else:
          break #物体变形太严重了，不用管

      temp=np.argmax(overlap)
      overlap[temp]=0
      index=lap_index[temp]
      #pdb.set_trace()
      try:#是否是已经碰撞的物体
        len(index)
        true_index=self.col_index.index(index) #找到在col_mask中对应的索引
        area+=np.sum(self.col_mask[true_index].astype(int))
        for single in index:
          indexs.append(single)
      except(TypeError):
        area+=self.last_label_area[index]
        indexs.append(index)
    

    return indexs
    
    

        

      # indexs=np.where(color_mask==True)[0]
      # for index in indexs:
      #   newmask=deepcopy(color_mask)
      #   newmask[index]=False #防止自己和自己相
      #   #看上一帧哪些object和这一帧重叠最大，主要看坐标
      #   overlap=self.last_labels[newmask]

      #   ind = np.where(abs(self.ini_label_area[color_mask]+self.ini_label_area[last]-label_area[i])<20)
      # new = np.where(color_mask==True)[0][ind]
    
  def update(self,image,j):
    new_colors=[]
    boxes=[]
    check=[]
    remember=[]
    num_labels,labels,label_color,label_area=connectedComponents_color(image)
    if(num_labels==self.num_labels): #没有出现两个同颜色物体相撞的情况
      self.collade=False
      self.col_index=[]
      self.col_mask=[]
    if(num_labels!=self.num_labels): print('numbersum error')
    #如果是black ，说明是静态，找重合即可
    for i in range(num_labels):
      if(j==18 and i==6): pdb.set_trace()
    #   #TODO:
    #   if(i==5 and label_area[i]==972): 
    #     for k in range(num_labels):
    #         mask=labels == k
    #         output = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
    #         output[:,:,:]=254
    #         output[:, :, 0][mask] = np.random.randint(0, 255)
    #         output[:, :, 1][mask] = np.random.randint(0, 255)
    #         output[:, :, 2][mask] = np.random.randint(0, 255)
    #         cv.imwrite('new{}.jpg'.format(k),output)
    #     pdb.set_trace()
      color_mask=self.ini_labels_color==label_color[i]
      len_color_mask=(color_mask.astype(int)).sum()
      mask=labels==i
      box=find_box(mask)
      boxes.append(box)
      index_collade=np.empty(0)
      if(label_color[i]==1): #black is static ,so just find the coordinate pixels
        ind = np.where(np.array([(self.ini_boxes[color_mask][j]==box).all() for j in range(len_color_mask)])) #index means the same color and the same location(which includes same area)
        index = np.where(color_mask==True)[0][ind]
      else:
        ind = np.where( (abs(self.last_label_area[color_mask]-label_area[i])<40)) #index means the same color and the same area
         #TODO:如果此帧中A和C碰撞，而刚好和B的大小相同，就会被映射到B上。
         
        #pdb.set_trace()
        if(ind[0].size==0):#TODO: 主要针对变形严重的长方体
          ind = np.where(np.sum(abs(np.vstack((np.mean(self.last_boxes[color_mask][:,:,0],axis=1),np.mean(self.last_boxes[color_mask][:,:,1],axis=1)))-np.vstack((np.mean(box[:,0]),np.mean(box[:,1])))),axis=0)<10) 
        #the width and the height of the box should be coherent
          h=math.sqrt(pow((box[0][0]-box[1][0]),2)+pow((box[0][1]-box[1][1]),2))
          w=math.sqrt(pow((box[1][0]-box[2][0]),2)+pow((box[1][1]-box[2][1]),2))
          if(w>h): h,w=w,h
          for ite,lastbox in enumerate(self.last_boxes[color_mask][ind]):
            lh=math.sqrt(pow((lastbox[0][0]-lastbox[1][0]),2)+pow((lastbox[0][1]-lastbox[1][1]),2))
            lw=math.sqrt(pow((lastbox[1][0]-lastbox[2][0]),2)+pow((lastbox[1][1]-lastbox[2][1]),2))
            if(lw>lh): lh,lw=lw,lh
            if(abs(h-lh)<0.2*lh and abs(w-lw)<0.2*lw): continue
            else:
              ind=np.delete(ind[0],ite)
              #pdb.set_trace() 
        index = np.where(color_mask==True)[0][ind]   
        if(self.collade==True):
          col_color_mask=np.array(self.col_color)==label_color[i]
          #pdb.set_trace()
          ind_collade = np.where((abs(self.col_area[col_color_mask]-label_area[i])<40))
          if(ind_collade[0].size==0):#TODO: 主要针对变形严重的长方体
            try:
              ind_collade = np.where(np.sum(abs(np.vstack((np.mean(np.array(self.col_boxes)[col_color_mask][:,:,0],axis=1),np.mean(np.array(self.col_boxes)[col_color_mask][:,:,1],axis=1)))-np.vstack((np.mean(box[:,0]),np.mean(box[:,1])))),axis=0)<10) 
            except(TypeError): pdb.set_trace()
          #the width and the height of the box should be coherent
            h=math.sqrt(pow((box[0][0]-box[1][0]),2)+pow((box[0][1]-box[1][1]),2))
            w=math.sqrt(pow((box[1][0]-box[2][0]),2)+pow((box[1][1]-box[2][1]),2))
            if(w>h): h,w=w,h
            #pdb.set_trace()
            for ite,lastbox in enumerate(np.array(self.col_boxes)[col_color_mask][ind_collade]):
              lh=math.sqrt(pow((lastbox[0][0]-lastbox[1][0]),2)+pow((lastbox[0][1]-lastbox[1][1]),2))
              lw=math.sqrt(pow((lastbox[1][0]-lastbox[2][0]),2)+pow((lastbox[1][1]-lastbox[2][1]),2))
              if(lw>lh): lh,lw=lw,lh
              if(abs(h-lh)<0.2*lh and abs(w-lw)<0.2*lw): continue
              else:
                ind_collade=np.delete(ind_collade[0],ite)
          try:
            index_collade = np.where(col_color_mask==True)[0][ind_collade] 
          except(IndexError):
            pdb.set_trace()
          
         
      
      flag=False
      #len(index)>0 and len(index_collade)==0: flag=false
      #len(index)==0 and len(index_collade)>0: flag=true
      #如果两个都大于0，就不确定
      #如果两个都等于0， 就得碰撞判断
      

      if(len(index)>0 or len(index_collade)>0): #只记录不会发生碰撞的物体的位置等
        #print('judge for the confusing case')
        index,flag=self.fitness(index,index_collade,box,flag)
        if(flag==False):
          check.append(index[0])
          self.last_boxes[index[0]]=box
          self.last_labels[mask]=index[0]
          self.last_label_area[index[0]]=label_area[i]
          self.last_labels_color[index[0]]=label_color[i]
        else:
          loc=index[0]
          #pdb.set_trace()
          check.append(self.col_index[loc])
          self.col_mask[loc]=mask
          self.col_boxes[loc]=box
          self.col_area[loc]=label_area[i]
          self.col_color[loc]=label_color[i]


        #   print(i,label_color[i],index)
        #   #pdb.set_trace()
        #   index=self.fitness(index,box)
        #  # pdb.set_trace()
        # if(len(index_collade)>0):#最复杂，这个物体有可能是碰撞过的也有可能不是
        #   print('judge for the confusing case with collision')
        #   print(i,label_color[i],index)
        #   #pdb.set_trace()
        #   index,flag=self.col_fitness(index,index_collade,box,flag)
        # if(len(index)==0 and )
        #pdb.set_trace()
      else:
        print('lose the object')
        #ocasion1: two balls of same color touch each other
        if(num_labels!=self.num_labels):#这里默认了碰撞的物体不会找到对应的原始标签
          print('two things of the same color touch each other')
          #pdb.set_trace()
          index=self.collision(label_area,i,label_color[i],mask)
          #pdb.set_trace()
          #for index in indexs:
          if(len(index)!=0):
            self.collade=True
            check.append(index)
            try:
              loc=self.col_index.index(index)
              self.col_mask[loc]=mask
              self.col_boxes[loc]=box
              self.col_area[loc]=label_area[i]
              self.col_color[loc]=label_color[i]
              
            except(ValueError):
              self.col_mask.append(mask)
              self.col_index.append(index)
              self.col_boxes.append(box)
              self.col_area=np.append(self.col_area,label_area[i])
              self.col_color=np.append(self.col_color,label_color[i])
          else:
            print('cannot detect the collision')
            pdb.set_trace()
        else:
          print('ERROR')
          print(i,label_color[i],box,label_area[i])
          print(self.last_label_area)
          print(self.last_labels_color)

          pdb.set_trace()
      new_colors.append(self.colors[index[0]])


      

    print('--check---')
    print(check)
    # for i in range(len(check)):
    #   try:
    #     len(check[i])
        
    #   except(TypeError):

    self.check=check

      #pdb.set_trace()
    return new_colors,boxes,check
        #还有可能是空数组

     

def create_gif(image_list,gif_name,duration=0.2):
    frames=[]
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name,frames,'GIF',duration=duration)





 
trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT','new']
 
def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]: 
    tracker = cv.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv.TrackerCSRT_create()
  elif trackerType == trackerTypes[8]:
    tracker = my_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
     
  return tracker


#first of all,you should specifically initialize the scene without any input
#load the initial scene
# print(color_to_id('white'))
# print(shared_if.Color)
#pdb.set_trace()
# Set video to load
for i in range(4,6):
  gif_name='TracGif/{}.gif'.format(i)
  image_list=[]
  for j in range(1,48):
    image_list.append('tracker_new_{}/MultiTracker{}.jpg'.format(i,j))
  create_gif(image_list,gif_name,duration=1.0)
pdb.set_trace()
for i in range(3,4):
  videoPath = "pic/try{}.gif".format(i)
  # Create a video capture object to read videos
  cap = cv.VideoCapture(videoPath)
  
  # Read first frame
  
  success, frame = cap.read()
  
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)
  #picpath='try1.png'
  #frame = frame[59:425,150:502]
  frame = frame[61:423,145:510]
  cv.imwrite('caps/frame{}.png'.format(i),frame)
  #366*352*3
  #pdb.set_trace()
  #cv.imwrite(picpath,frame)
  # with open("featurized_objects.csv","r") as csvfile:
  #     reader = csv.reader(csvfile)
  #     lines=[line for line in reader]
  #     pdb.set_trace()
  #     #TODO:
  im1,num_labels, labels,labels_color,label_area,colors,boxes=capture_new(frame)
  #pdb.set_trace()
  cv.imwrite('caps/cap{}.png'.format(i),im1)
  #pdb.set_trace()
  # Specify the tracker type
  trackerType = 'new'
  tracker=createTrackerByName(trackerType)
  #print(tracker)
  tracker.create(num_labels, labels,labels_color,label_area,boxes,colors)

  # Create MultiTracker object
  
  path='tracker_{}_{}'.format(trackerType,i)
  os.makedirs(path)
  # Initialize MultiTracker 
  # multiTracker = cv.MultiTracker_create()
  # for box in boxes:
  #     multiTracker.add(createTrackerByName(trackerType), frame, box)

  # Process video and track objects
  j=0
  while cap.isOpened():
      success, frame = cap.read()
      try:
        frame = frame[61:423,145:510]
      except:
        print(j)
        break
      j+=1
      cv.imwrite('{}/UnTrack{}.jpg'.format(path,j), frame)
      #pdb.set_trace()
      if not success:
          break
      # get updated location of objects in subsequent frames
      #TODO:
      #if(j==6): pdb.set_trace()
      #if(j==17): pdb.set_trace()
      new_colors, boxes, check = tracker.update(frame,j)

      # draw tracked objects
      frame=drawRectangles(boxes,frame,new_colors)
      frame=putnum(boxes,check,frame)

      # show frame
      cv.imwrite('{}/MultiTracker{}.jpg'.format(path,j), frame)


