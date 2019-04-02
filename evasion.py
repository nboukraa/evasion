# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:18:52 2019

@author: cyberchinois
"""

import cv2
import matplotlib.pyplot as plt

path = "C:\Users\cyberchinois\Desktop\data\"
k=1
f = open('list.txt', 'rt')
for line in f:
    w = line.split()[0]
    image = cv2.imread(w)
    plt.figure(0)
    plt.imshow(image)    
    f.close()
    k=k+1
    if k == 10:
        break