import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
import csv
import re  
import pandas as pd
pattern = re.compile(r'\d+')
def draw_circle(circles,k):
    fig = plt.figure(dpi=256)
    for c in circles:
        circle = plt.Circle(c[0], c[1], color='y', fill='y')
        plt.gcf().gca().add_artist(circle)

    plt.axis('equal')
    
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    pdb.set_trace()
    plt.savefig('result/{}.jpg'.format(int(k)))
#draw_circle((((0.35*256,0.22923309*256),0.05859375*256),((0.75*256,0.26048306*256),0.12109375*256),((0.4140625*256,0.71478*256),0.015625*256)))


# 0.35,0.22923309,0.0,0.05859375,BALL,GREEN
# 0.75,0.26048306,0.0,0.12109375,BALL,BLUE
# 0.4140625,0.71478,0.0,0.015625,BALL,RED

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
                li.append([x,y])
            draw_circle(li,i-5)
            pdb.set_trace()
            l.append(li)
    

