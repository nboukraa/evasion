# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:59:37 2019

@author: cyberchinois
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

path = r"C:\Users\cyberchinois\Desktop\data\\"

os.chdir(path)
RepImages=(path)

dbRGB=[]
dbHSI=[]
dgris=[]
k=1

#def load_image():
f = open(RepImages + 'listfull.txt', 'rt')
for line in f:
   dbRGB.append(cv2.imread(RepImages+line.split()[0]))
   dbHSI.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2HSV))
   dgris.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2GRAY))   
   k=k+1
   if k== 20:
       break
f.close()

#load_image()
niv_gris_avg = []
for i in dgris:
    niv_gris_avg.append(dgris.mean())

plt.figure()
plt.plot(niv_gris_avg)
plt.show()