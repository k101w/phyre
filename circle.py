import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
import csv
import re  
import pandas as pd
pattern = re.compile(r'\d+')
def draw_circle(circles,k):
    if(k==114): pdb.set_trace()
    diameter=[0.05859375*128,0.12109375*128,0.015625*128]
    color=['g','b','r']
    fig = plt.figure(dpi=256)
    i=0
    for c in circles:
        circle = plt.Circle(tuple(c), diameter[i], color=color[i], fill=color[i])
        #pdb.set_trace()
        i+=1
        plt.gcf().gca().add_artist(circle)

    plt.axis('scaled')
    
    plt.xlim(0, 256).set_visible(False)
    plt.ylim(0, 256).set_visible(False)
    plt.savefig('result/{}.jpg'.format(int(k)))
#draw_circle((((0.35*256,0.22923309*256),0.05859375*256),((0.75*256,0.26048306*256),0.12109375*256),((0.4140625*256,0.71478*256),0.015625*256)))


# 0.35,0.22923309,0.0,0.05859375,BALL,GREEN
# 0.75,0.26048306,0.0,0.12109375,BALL,BLUE
# 0.4140625,0.71478,0.0,0.015625,BALL,RED


# 0.06576842,0.029414058,0.37352723,0.05859375,BALL,GREEN
# 0.84858316,0.060664047,0.7646914,0.12109375,BALL,BLUE
# 0.22582778,0.07440075,0.9898972,0.1484375,BALL,RED
l = []
with open('exp_local/2022.03.08/08_29_08/fold0/act5/pos930.csv','rt') as f: 
   cr = csv.reader(f)
   i=0
   for row in cr:
        i+=1
        if(i>=5):
            li=[]
            for r in row:
                a,b,c,d = pattern.findall(r)
                x=float(str(a)+'.'+str(b))*256
                y=float(str(c)+'.'+str(d))*256
                #pdb.set_trace()
                li.append([x,y])
            draw_circle(li,i-5)
            l.append(li)
    

