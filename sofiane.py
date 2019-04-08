# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:16:52 2019

@author: SK
"""

import os
import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#os.chdir('data/')
#RepImages=('data/)

def cadre (x):
    res =x
    res[10:246,10:310,:]=0
    return res

def lisser(x):
    kernel = np.ones((5,5),np.float32)/25
    #dst = cv2.filter2D(x,-1,kernel)
    dst =cv2.bilateralFilter(x,9,75,75)
    return(x)
    
def myWriteCSV(aFullFileName,aListeResultats):
#Inputs : aFullFileName - nom du fichier à créer avec le full path
#       : aListeResults - liste avec les numeros à ecrire
    
    with open(aFullFileName, 'w') as output:
        [output.write(str(x) + "\n") for x in aListeResultats]
    output.close()
    
def Coeff_De_Correlation (img,template,method):
    method = eval(method)
#['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return min_val,max_val,top_left


def laplacian (x):
    laplacian = cv2.Laplacian(x,cv2.CV_64F)
    laplacian=laplacian.astype(np.uint8)
    return laplacian

def gradient (x):
    sobelx = cv2.Sobel(x,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(x,cv2.CV_64F,0,1,ksize=5)
    x2=sobelx**2
    y2=sobely**2
    res=np.sqrt(x2+y2)
    maxv=np.max(res)    
    if (maxv==0):
        maxv=1
    res=res[:]*255/maxv
    res=res.astype(np.uint8)
    return res

def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x)*h
    plt.figure(figsize=(20,20))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')

def myEvaluation(aRep,aListeResultats):
#Inputs : aRep - full path
#       : aListeResultats - liste generique avec 1 seule dimension qui contient notre prediction
#Output : precision et recall

    #lecture
    GoldList=[]
    with open(aRep + 'goldResult.csv', encoding="utf-8") as f:
        for line in f:
            
            GoldList.append(int(line)-1)  # car les images commencent à 1 et python à 0 
    f.close
    
    #initialisation
    true_pos = 0
    false_pos = 0
    false_neg = 0
    
    #evaluation
    for key in aListeResultats:
        if key in GoldList:
            true_pos += 1
        else:
            false_pos += 1
            print("false pose "+repr(key))
    
    for key in GoldList:
        if key not in aListeResultats:
            false_neg += 1
            print("false negative "+repr(key))
    
    #calcul indicateurs
    if true_pos + false_pos != 0:
        precision = float(true_pos) / (true_pos + false_pos) * 100.0
    else:
        precision = 0.0
    
    if true_pos + false_neg != 0:
        recall = float(true_pos) / (true_pos + false_neg + false_pos) * 100.0
    else:
        recall = 0.0
        
    return precision, recall,true_pos,false_pos,false_neg     
    
dbRGB=[]
dbHSI=[]
dgris=[]
dgreen=[]
mycadre=[]
f = open('data/listfull.txt', 'rt')

j=0
for line in f:
   #dbRGB.append(cv2.imread('data/'+line.split()[0]))
   #x=dbRGB[j][:,:,2]
   #print(j)
   # dgreen.append(gradient(x))
   #dbHSI.append(cv2.imread('data/'+line.split()[0],cv2.COLOR_BGR2HSV))
   x=cv2.imread('data/'+line.split()[0],cv2.COLOR_BGR2GRAY)
   #x=gradient(x)
   #print(j)
   dgris.append(lisser(x))
   j=j+1
f.close()
"""
mycadreH=[x [0:40,:] for x in dgreen]
mycadreB=[x [-40:,:,1] for x in dgris]
mycadreD=[x [:,-40:,1] for x in dgris]
mycadreG=[x [:,0:40] for x in dgreen]


myavgH=[np.mean(x) for x in mycadreH]
myavgG=[np.mean(x) for x in mycadreG]
myavgB=[np.mean(x) for x in mycadreB]
myavgD=[np.mean(x) for x in mycadreD]
"""
def Loc_sub_img (img,template,method):
    method = eval(method)
#['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    # Apply template Matching
    res = cv2.matchTemplate(img,template,method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return min_val, max_val,top_left

def VIndexation1 (ListeImage,Methode_grad,Methode_correlation,CoefDiff1):
    Resultats=[]
    Resultatsdet=[]
    grad=eval(Methode_grad)
    for i in range (0,len(ListeImage)-2):
        im1=grad (ListeImage[i])#gradient(dgris[i]) laplacian
        im2=grad (ListeImage[i+1])#gradient(dgris[i+1])
        #x=x.astype(int)
        #print(x)
        #print(dgris[i])
        (m,M,t) = Coeff_De_Correlation (im1,im2,Methode_correlation)

        if(M<CoefDiff1):
            s =repr(i+2) + ' ' + repr(M)
            Resultats.append(i+1)
            Resultatsdet.append(s)
    return Resultatsdet,Resultats

def VIndexation2 (ListeImage,Methode_grad,Methode_correlation,CoefDiff1,CoeffDiff2=0):
    Resultats=[]
    Resultatsdet=[]
    grad=eval(Methode_grad)
    for i in range (0,len(ListeImage)-2):
        im1=grad (ListeImage[i])#gradient(dgris[i]) laplacian
        im2=grad (ListeImage[i+1])#gradient(dgris[i+1])

        (m,M,t) = Coeff_De_Correlation (im1,im2,Methode_correlation)
        
        if(M<CoefDiff1):
            #M_2=max(M1,M2,M3,M4)
            im3=grad (ListeImage[i+2])
            (m2,M2,t2) = Coeff_De_Correlation (im2,im3,Methode_correlation)
            if(M2>CoeffDiff2):
                s =repr(i+2) + ' ' + repr(M)+ ' ' + repr(M2)
                Resultats.append(i+1)
                Resultatsdet.append(s)
    return Resultatsdet,Resultats

def VIndexation3 (ListeImage,Methode_grad,Methode_correlation,CoefDiff1,CoeffDiff2=0):
    Resultats=[]
    Resultatsdet=[]
    grad=eval(Methode_grad)
    #Resultats.append(1)
    rows,columns,d=ListeImage[0].shape
    rows=int(rows/4)
    columns=int(columns/4)
    size_blockx=70
    size_blocky=80
    for i in range (0,len(ListeImage)-2):
        im1=grad (ListeImage[i])#gradient(dgris[i]) laplacian
        im2=grad (ListeImage[i+1])#gradient(dgris[i+1])

        subimg1=im1[rows:rows+size_blockx,columns:columns+size_blocky,:]
        subimg2=im1[rows:rows+size_blockx,3*columns:3*columns+size_blocky,:]
        subimg3=im1[3*rows:3*rows+size_blockx,columns:columns+size_blocky,:]
        subimg4=im1[3*rows:3*rows+size_blockx,3*columns:3*columns+size_blocky,:]
        (m1,M1,t1) = Loc_sub_img(subimg1,im2,Methode_correlation)
        (m2,M2,t2) = Loc_sub_img(subimg2,im2,Methode_correlation)
        (m3,M3,t3) = Loc_sub_img(subimg3,im2,Methode_correlation)
        (m4,M4,t4) = Loc_sub_img(subimg4,im2,Methode_correlation)
        #(m,M,t) = Coeff_De_Correlation (im1,im2,Methode_correlation)
        
        M=(M1+M2+M3+M4)/4

        if(M<CoefDiff1):

            im3=gradient(ListeImage[i+2])#gradient(dgris[i])
            subimg1=im2[rows:rows+size_blockx,columns:columns+size_blocky,:]
            subimg2=im2[rows:rows+size_blockx,3*columns:3*columns+size_blocky,:]
            subimg3=im2[3*rows:3*rows+size_blockx,columns:columns+size_blocky,:]
            subimg4=im2[3*rows:3*rows+size_blockx,3*columns:3*columns+size_blocky,:]
            
            (mm1,MM1,tt1) = Loc_sub_img(subimg1,im3,Methode_correlation)
            (mm2,MM2,tt2) = Loc_sub_img(subimg2,im3,Methode_correlation)
            (mm3,MM3,tt3) = Loc_sub_img(subimg3,im3,Methode_correlation)
            (mm4,MM4,tt4) = Loc_sub_img(subimg4,im3,Methode_correlation)
            
            M_2=(MM1+MM2+MM3+MM4)/4

            if(M_2>CoeffDiff2):
                s =repr(i+2) + ' ' + repr(M)+ ' ' + repr(M_2)
                Resultats.append(i+1)
                Resultatsdet.append(s)
    return Resultatsdet,Resultats

#rd,r=VIndexation3(dgris,'gradient','cv2.TM_CCORR_NORMED',0.75,0.65)
rd,r=VIndexation1(dgris,'gradient','cv2.TM_CCORR_NORMED',0.75)
myWriteCSV('resultats.txt',rd)  

precision,recall,true_pos,false_pos,false_neg =myEvaluation("",r)

print("=============================")
print(precision)
print(recall)
print(true_pos)
print(false_pos) 
print(false_neg)     
print("=============================")
