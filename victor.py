# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:13:32 2019

@author: costav
"""

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.chdir('C:\\Users\\costav\\Documents\\big data\\Ecole\\Multimedia\\TP2\\')
RepImages=('C:\\Users\\costav\\Documents\\big data\\Ecole\\Multimedia\\data\\')

dbRGB=[]
dbHSI=[]
dbgris=[]
f = open(RepImages + 'listfull.txt', 'rt')
for line in f:
    dbRGB.append(cv2.imread(RepImages+line.split()[0]))
    dbHSI.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2HSV))
    dbgris.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2GRAY))
f.close()

myavg=[np.mean(x) for x in dbgris]

plt.figure()
plt.plot(myavg)
plt.show()



#cv2.imread('zebres-small.tif',  cv2.COLOR_BGR2GRAY)
#rotation de l'image de 45 degrés

#rows,cols = Image.shape

#M = cv2.getRotationMatrix2D((rows/2,cols/2),45,1)
#Image = cv2.warpAffine(grayImage,M,(cols,rows))

#plt.figure(0)
#plt.subplot(2,2,1)
#plt.imshow(grayImage)
#plt.title('image originale')
#plt.subplot(2,1,2)
#plt.imshow(destImage)
#plt.title('image tournee')

#qu'observez-vous ?

#m = numpy.moyenne(grayImage)
#mr = numpy.moyenne(destImage)
#v= numpy.variance(grayImage)
#vr = numpy.variance(destImage)

#plt.figure(1)
#x=[m, mr]
#y=[v, vr]
#plt.plot(m, v, '.')
#plt.annotate('orig', [m,v])
#plt.plot(mr, vr, '.')
#plt.annotate('tournee', [mr,vr])
#plt.title('exploration de l''espace des attributs')

