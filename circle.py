import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
import csv
import re  
import pandas as pd
import os
import imageio
pattern = re.compile(r'\d+')

def draw_circle(circles,k):
    diameter=[0.05859375*128,0.12109375*128,0.0234375*128]
    color=['g','b','r']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    i=0
    for c in circles:
        circle = plt.Circle(tuple(c), diameter[i], color=color[i], fill=color[i])
        #pdb.set_trace()
        i+=1
        plt.gcf().gca().add_artist(circle)

    ax.set_aspect('equal', adjustable='box')
    
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.savefig('exp_local/2022.03.10/05_31_44/fold0/act9/figs/{}.jpg'.format(int(k)))


def gif(path, duration=0.01):
    frames=[]
    length=len(os.listdir(path))

    for i in range(length):
        image_name=os.path.join(path,'{}.jpg'.format(i))
        frames.append(imageio.imread(image_name))

    imageio.mimsave('exp_local/2022.03.10//05_31_44/fold0/act9/act9.gif',frames,'GIF',duration=duration)

#draw_circle((((0.35*256,0.22923309*256),0.05859375*256),((0.75*256,0.26048306*256),0.12109375*256),((0.4140625*256,0.71478*256),0.015625*256)))


# 0.35,0.22923309,0.0,0.05859375,BALL,GREEN
# 0.75,0.26048306,0.0,0.12109375,BALL,BLUE
# 0.4140625,0.71478,0.0,0.015625,BALL,RED


# 0.06576842,0.029414058,0.37352723,0.05859375,BALL,GREEN
# 0.84858316,0.060664047,0.7646914,0.12109375,BALL,BLUE
# 0.22582778,0.07440075,0.9898972,0.1484375,BALL,RED



l=[]
with open('exp_local/2022.03.10/05_31_44/fold0/act9/pos990.csv','rt') as f: 
   cr = csv.reader(f)
   i=0
   for row in cr:
        i+=1
        if(i>=3):
            li=[]
            for r in row:
                a,b,c,d = pattern.findall(r)
                x=float(str(a)+'.'+str(b))*256
                y=float(str(c)+'.'+str(d))*256
                #pdb.set_trace()
                li.append([x,y])
            draw_circle(li,i-3)
            l.append(li)
    
path='exp_local/2022.03.10/05_31_44/fold0/act9/figs'
gif(path)
