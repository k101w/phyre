import numpy as np
import pdb
import matplotlib.pyplot as plt
import torch
def draw_circle(circles):
    fig = plt.figure(dpi=256)
    for c in circles:
        circle = plt.Circle(c[0], c[1], color='y', fill='y')
        plt.gcf().gca().add_artist(circle)

    plt.axis('equal')
    
    plt.xlim(0, 256)
    plt.ylim(0, 256)

    plt.show()
    plt.savefig('1.jpg')
draw_circle((((0.35*256,0.22923309*256),0.05859375*256),((0.75*256,0.26048306*256),0.12109375*256),((0.4140625*256,0.71478*256),0.015625*256)))


# 0.35,0.22923309,0.0,0.05859375,BALL,GREEN
# 0.75,0.26048306,0.0,0.12109375,BALL,BLUE
# 0.4140625,0.71478,0.0,0.015625,BALL,RED