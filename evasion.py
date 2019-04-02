# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:18:52 2019

@author: cyberchinois
"""

import cv2
import matplotlib.pyplot as plt
import os


path = r"C:\Users\cyberchinois\Desktop\data\\"

os.chdir(path)

images = []

k=1
f = open('list.txt', 'rt')
for line in f:
    w = line.split()[0]
    image = cv2.imread(w)
    images.append(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)       
    k=k+1
    if k == 10:
        break
f.close()