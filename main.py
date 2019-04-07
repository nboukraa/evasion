# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:59:37 2019

@author: cyberchinois
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def myWriteCSV(aFullFileName,aListeResultats):
    
    with open(aFullFileName, 'w') as output:
        [output.write(str(x) + "\n") for x in aListeResultats]
    output.close()


def myLoadImage(aFullPath,aFileName,aTypeColor='gris',abreak=None):

    lDecomposition=[]
    lcount=1
    f = open(aFullPath + aFileName, 'rt')
    
    #ecrit comme ça pour minimiser des calculs/evaluations au sein d'une boucle
    if aTypeColor == 'RGB':
        for line in f:
           lDecomposition.append(cv2.imread(aFullPath+line.split()[0]))
           lcount+=1
           if lcount == abreak:
               break
    
    if aTypeColor =='HSV':
        for line in f:
           lDecomposition.append(cv2.imread(aFullPath+line.split()[0],cv2.COLOR_BGR2HSV))
           lcount+=1
           if lcount == abreak:
               break
         
    if aTypeColor =='gris':
        for line in f:
           lIm=cv2.imread(aFullPath+line.split()[0])
           lDecomposition.append(cv2.cvtColor(lIm,cv2.COLOR_BGR2GRAY))
           lcount+=1
           if lcount == abreak:
               break
  
    f.close()
    return lDecomposition


def myEvaluation(aRep,aListeResultats):
    
    #lecture
    GoldList=[]
    with open(aRep + 'goldResult.csv', encoding="utf-8") as f:
        for line in f:
            GoldList.append(int(line)-1)
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
    
    for key in GoldList:
        if key not in aListeResultats:
            false_neg += 1
    
    #calcul indicateurs
    if true_pos + false_pos != 0:
        precision = float(true_pos) / (true_pos + false_pos) * 100.0
    else:
        precision = 0.0
    
    if true_pos + false_neg != 0:
        recall = float(true_pos) / (true_pos + false_neg + false_pos) * 100.0
    else:
        recall = 0.0
        
    return precision, recall


def myLissage(aList,aNbImagesLissage=5):
    #utiliser nombre impair
    ListLisse=[]
    
    #first images
    for i in range(0,int((aNbImagesLissage-1)/2)):
        ListLisse.append(aList[i]) 
    
    for i in range(1,len(aList)-int((aNbImagesLissage-1)/2)):
        valeur=0.0    
        for j in range(int(-(aNbImagesLissage-1)/2),int((aNbImagesLissage-1)/2+1)):
            valeur+=aList[i+j]
        ListLisse.append(valeur/aNbImagesLissage)
    
    #last images
    for i in range(-int((aNbImagesLissage-1)/2),0):
        ListLisse.append(aList[i]) 

    return ListLisse


def myCutOff(aCutOff,aList):
    
    lResult=[]
    for i in range(len(aList)):
        if abs(aList[i]) > aCutOff:
            lResult.append(i)
    return lResult


def myDivideBlocks(aNbBlocks,aImage):

    (dimYpix, dimXpix) = aImage.shape

    dimYpixBlock=int(dimYpix/aNbBlocks)
    dimXpixBlock=int(dimXpix/aNbBlocks)    
    
    liste_block_frame=[]
    for bl in range(0,nbBlocks**2) : liste_block_frame.append(aImage
                   [(bl//nbBlocks)*dimYpixBlock : (bl//nbBlocks+1)*dimYpixBlock,
                    (bl%nbBlocks)*dimXpixBlock :(bl%nbBlocks+1)*dimXpixBlock])

    return liste_block_frame

### MAIN CODE ###

#path = r"C:\Users\cyberchinois\Desktop\data\\"
path = r"C:\Users\costav\Documents\big data\Ecole\Multimedia\data\\"
#path = r"C:\Users\vicks\Documents\Ecole\Multimedia\data\\"
#RepCode=r"C:\Users\vicks\Documents\Ecole\Multimedia\evasion\\"
os.chdir(path)
RepImages=(path)

RepCode=r"C:\Users\costav\Documents\big data\Ecole\Multimedia\evasion\\"

nbBlocks=4 #pour 4x4
liste_im=myLoadImage(RepImages,'listfull.txt','gris')

NbImages=len(liste_im)

liste_block_images=[]
for im in range(NbImages):
    liste_block_images.append(myDivideBlocks(nbBlocks,liste_im[im]))
    
lhistblock=[]
for iblock in range(nbBlocks**2):
    liste=[]
    for im in range(NbImages):
        block_hist=cv2.calcHist(liste_block_images[im][iblock],[0],None,[64],[0,255])
        liste.append(cv2.normalize(block_hist,block_hist, 1.0, 0.0, cv2.NORM_L1))
    lhistblock.append(liste)

# Calcul de la distance Chi2, plus adapté aux histogrammes
lChiDistBloc=[]
for iblock in range(nbBlocks**2):
    dist=[]
    #dist.append(0) #initialisation #PAS BESOIN
    for im in range(NbImages):
        dist.append(cv2.compareHist(lhistblock[iblock][im],lhistblock[iblock][im-1],cv2.HISTCMP_CHISQR_ALT))
    lChiDistBloc.append(dist)

#zero false negatives
seuil1=1.5
seuil2=0.65

myResult=[]

for im in range(NbImages): 
    blockcounteur=0
    for iblock in range(nbBlocks**2):
        if lChiDistBloc[iblock][im]>seuil1:
            blockcounteur+=1
    if float(blockcounteur)/float(nbBlocks**2)>=seuil2:
        myResult.append(im)
        
precision, recall = myEvaluation(RepCode,myResult)
print(precision)
print(recall)
myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom   
    