# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:59:37 2019

@author: cyberchinois
"""

import os
import cv2
#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import csv


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
           lDecomposition.append(cv2.imread(aFullPath+line.split()[0],cv2.COLOR_BGR2GRAY))   
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
            GoldList.append(int(line))
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


#    aRGB=[]
#    aHSV=[]
#    agris=[]
#    lcount=1
#    f = open(aFullPath + aFileName, 'rt')
#    for line in f:
#       aRGB.append(cv2.imread(aFullPath+line.split()[0]))
#       aHSV.append(cv2.imread(aFullPath+line.split()[0],cv2.COLOR_BGR2HSV))
#       agris.append(cv2.imread(aFullPath+line.split()[0],cv2.COLOR_BGR2GRAY))   
#       lcount+=1
#       if lcount == abreak:
#           break
#    f.close()
#    return aRGB, aHSV, agris


### MAIN CODE ###

#path = r"C:\Users\cyberchinois\Desktop\data\\"
#path = r"C:\Users\costav\Documents\big data\Ecole\Multimedia\data\\"
path = r"C:\Users\vicks\Documents\Ecole\Multimedia\data\\"
RepCode=r"C:\Users\vicks\Documents\Ecole\Multimedia\evasion\\"
os.chdir(path)
RepImages=(path)

#RepCode=r"C:\Users\costav\Documents\big data\Ecole\Multimedia\evasion\\"

#load
dbgris = myLoadImage(RepImages,'listfull.txt','gris',20)
dbRGB = myLoadImage(RepImages,'listfull.txt','RGB',20)
dbHSV = myLoadImage(RepImages,'listfull.txt','HSV',20)

#treatment 1
niv_gris_avg =[np.mean(x) for x in dbgris]


#visu
plt.axis([0, 100, 0, 170])
plt.title('Niveaux de gris sans lissage')
plt.plot(niv_gris_avg)
plt.show()

#lissage de la moyenne (utiliser nombre impair)
niv_gris_avg_liss=myLissage(niv_gris_avg,5)

#visu
plt.axis([0, 500, 0, 170])
plt.title('Niveaux de gris sans lissage')
plt.plot(niv_gris_avg)
plt.plot(niv_gris_avg_liss)
plt.show()

#niv_gris_avg=niv_gris_avg_liss

#treatment 2
derive_niv_gris_avg=[]
derive_niv_gris_avg.append(0)
for i in range(1,len(niv_gris_avg)-1):
    derive_niv_gris_avg.append(niv_gris_avg[i]-niv_gris_avg[i-1])
derive_niv_gris_avg.append(0)

#graphs
plt.axis([0, 2500, -20, 20])
plt.title('Diff niveaux de gris')
plt.plot(derive_niv_gris_avg)
plt.show()

#histogramme
plt.hist(derive_niv_gris_avg,200)
plt.title('Histograme du niveau de gris')
plt.axis([-20, 20, 0, 80])
plt.show()

#filtre
myResult=myCutOff(10,derive_niv_gris_avg)

#ecriture
myWriteCSV(RepCode+'myResult.csv',myResult)
#evaluation
precision, recall = myEvaluation(RepCode,myResult)
print(precision,recall,sep=' ')


#evaluation en boucle
precision=[]
recall=[]

for i in range(1,50):
    myCutOff=i
    myResult=[]
    
    for i in range(len(derive_niv_gris_avg)):
        if abs(derive_niv_gris_avg[i]) > myCutOff:
            myResult.append(i)        
    
    lprecision, lrecall = myEvaluation(RepCode,myResult)
    precision.append(lprecision)
    recall.append(lrecall)

#graphs
plt.axis([0, len(precision), 0, max(max(precision),max(recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(precision)
plt.plot(recall)
plt.show()

#max
CutoffOptimal=precision.index(max(precision))
print(CutoffOptimal, max(precision), recall[CutoffOptimal])