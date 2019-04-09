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
import statistics as s

path = r"C:\Users\cyberchinois\Desktop\data\\"

os.chdir(path)
RepImages=(path)

dbRGB=[]
dbHSI=[]
dgris=[]
niv_gris_avg = []
k=1

#def load_image():
f = open(RepImages + 'listfull.txt', 'rt')
for line in f:
   dbRGB.append(cv2.imread(RepImages+line.split()[0]))
   dbHSI.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2HSV))
   dgris.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2GRAY))
   niv_gris_avg.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2GRAY).mean())
   k=k+1
   if k== 500:
       break
f.close()

#load_image()

#plt.figure()
#plt.plot(niv_gris_avg)
#plt.show()

print(s.mean(niv_gris_avg))
print(s.stdev(niv_gris_avg))

res_avg_gris = []
moyenne =round(s.mean(niv_gris_avg), 0)

for i in niv_gris_avg:    
    if i-moyenne > 10:
        res_avg_gris.append(niv_gris_avg.index(i))
        
print(niv_gris_avg)
print(res_avg_gris)