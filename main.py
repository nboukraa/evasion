# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:59:37 2019

@author: cyberchinois
"""

import os
import cv2
import pandas as pd
import numpy as np

path = r"C:\Users\cyberchinois\Desktop\data\\"

os.chdir(path)
RepImages=(path)

dbRGB=[]
dbHSI=[]
dgris=[]
f = open(RepImages + 'listfull.txt', 'rt')
for line in f:
   dbRGB.append(cv2.imread(RepImages+line.split()[0]))
   dbHSI.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2HSV))
   dgris.append(cv2.imread(RepImages+line.split()[0],cv2.COLOR_BGR2GRAY))
f.close()