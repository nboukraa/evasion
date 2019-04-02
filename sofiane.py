# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:16:52 2019

@author: SK
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 

import matplotlib.image as mpimg
import numpy as np
#img = mpimg.imread("monimage.png")


f = open('data/list.txt', 'rt')
i=0
img=np.zeros(shape=(3000,256,320,3))
for line in f:
    w = line.split()[0]
    img[i]= np.array(mpimg.imread("data/"+w))
    i=i+1
#dataImage = pd.DataFrame()
print(img[0].shape)

import matplotlib.pyplot as plt
plt.imshow(img[20][0])
plt.show(100)