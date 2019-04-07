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
import math


def myWriteCSV(aFullFileName, aListeResultats):
    # Inputs : aFullFileName - nom du fichier à créer avec le full path
    #       : aListeResults - liste avec les numeros à ecrire

    with open(aFullFileName, 'w') as output:
        [output.write(str(x + 1) + "\n") for x in aListeResultats]
    output.close()


def myLoadImage(aFullPath, aFileName, aTypeColor='gris', abreak=None):
    # Inputs : aFullPath - full path
    #       : aFileName - nom du fichier qui contient les images à lire
    #       : aTypeColor - type de structure de couleur souhaité (RGB,HSV ou gris)
    #       : abreak - limite maximale d'images à extraire
    # Output : lDecomposition - liste avec 3 dimensions (dimX, dimY, 3 channels pour la couleur)
    # Modif  : Costa 5/4/2019 - correction de l'extraction en gris pour avoir 1 seul channel

    lDecomposition = []
    lcount = 1
    f = open(aFullPath + aFileName, 'rt')

    # ecrit comme ça pour minimiser des calculs/evaluations au sein d'une boucle
    if aTypeColor == 'RGB':
        for line in f:
            lDecomposition.append(cv2.imread(aFullPath + line.split()[0]))
            lcount += 1
            if lcount == abreak:
                break

    if aTypeColor == 'HSV':
        for line in f:
            lDecomposition.append(cv2.imread(aFullPath + line.split()[0], cv2.COLOR_BGR2HSV))
            lcount += 1
            if lcount == abreak:
                break

    if aTypeColor == 'gris':
        for line in f:
            lIm = cv2.imread(aFullPath + line.split()[0])
            lDecomposition.append(cv2.cvtColor(lIm, cv2.COLOR_BGR2GRAY))
            lcount += 1
            if lcount == abreak:
                break

    f.close()
    return lDecomposition


def myEvaluation(aRep, aListeResultats):
    # Inputs : aRep - full path
    #       : aListeResultats - liste generique avec 1 seule dimension qui contient notre prediction
    # Output : precision et recall

    # lecture
    GoldList = []
    with open(aRep + 'goldResult.csv', encoding="utf-8") as f:
        for line in f:
            GoldList.append(int(line) - 1)  # car les images commencent à 1 et python à 0
    f.close

    # initialisation
    true_pos = 0
    false_pos = 0
    false_neg = 0

    # evaluation
    for key in aListeResultats:
        if key in GoldList:
            true_pos += 1
        else:
            false_pos += 1

    for key in GoldList:
        if key not in aListeResultats:
            false_neg += 1

    # calcul indicateurs
    if true_pos + false_pos != 0:
        precision = float(true_pos) / (true_pos + false_pos) * 100.0
    else:
        precision = 0.0

    if true_pos + false_neg != 0:
        recall = float(true_pos) / (true_pos + false_neg + false_pos) * 100.0
    else:
        recall = 0.0

    return precision, recall


def myLissage(aList, aNbImagesLissage=3):
    # Inputs : aList - liste generique avec 1 seule dimension qui contient notre prediction
    #       : aNbImagesLissage - sur combien d'images on realise la moyenne
    # Output : ListLisse - nouvelle liste avec le lissage

    # utiliser nombre impair
    ListLisse = []

    # first images
    for i in range(0, int((aNbImagesLissage - 1) / 2)):
        ListLisse.append(aList[i])

    for i in range(1, len(aList) - int((aNbImagesLissage - 1) / 2)):
        valeur = 0.0
        for j in range(int(-(aNbImagesLissage - 1) / 2), int((aNbImagesLissage - 1) / 2 + 1)):
            valeur += aList[i + j]
        ListLisse.append(valeur / aNbImagesLissage)

    # last images
    for i in range(-int((aNbImagesLissage - 1) / 2), 0):
        ListLisse.append(aList[i])

    return ListLisse


def myCutOff(aCutOff, aList):
    # Inputs : aCutoff - entier avec le niveau de seuil
    #       : aList - liste generique avec 1 seule dimension qui contient un des traiments
    # Output : lResult - liste avec les images qui ont été selectionées

    lResult = []
    for i in range(len(aList)):
        if abs(aList[i]) > aCutOff:
            lResult.append(i)
    return lResult


def myDivideBlocks(aNbBlocks, aImage):
    # Inputs : aNbBlocks - entier avec une dimension du carré (nb blocks=x^2)
    #       : aImage - image à decouper avec 1 seul channel
    # Output : liste_block_frame - liste avec les X blocks

    (dimYpix, dimXpix) = aImage.shape

    dimYpixBlock = int(dimYpix / aNbBlocks)
    dimXpixBlock = int(dimXpix / aNbBlocks)

    liste_block_frame = []
    for bl in range(0, aNbBlocks ** 2):
        liste_block_frame.append(aImage[(bl // aNbBlocks) * dimYpixBlock :
                                        (bl // aNbBlocks + 1) * dimYpixBlock,\
                                 (bl % aNbBlocks) * dimXpixBlock:(bl % aNbBlocks + 1) * dimXpixBlock])

    return liste_block_frame

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

#Gestion des paths pour garantir que ça ne pete pas.
#Si vous saviez faire ça correctement, merci de partager l'info

#path = r"C:\Users\cyberchinois\Desktop\data\\"
#path = r"C:\Users\costav\Documents\big data\Ecole\Multimedia\data\\"

# path = r"C:\Users\vicks\Documents\Ecole\Multimedia\data\\"
# RepCode=r"C:\Users\vicks\Documents\Ecole\Multimedia\evasion\\"

path = r"E:\Dropbox\CES DAta sciences\Cours\Module 4\Projet\data\\"
RepCode=r"E:\Dropbox\CES DAta sciences\Cours\Module 4\Projet\evasion\\"
os.chdir(path)
RepImages=(path)

#RepCode=r"C:\Users\costav\Documents\big data\Ecole\Multimedia\evasion\\"

#load
dbgris = myLoadImage(RepImages,'listfull.txt','gris')
NbImages=len(dbgris)

#traitement 1 : la moyenne
niv_gris_avg =[np.mean(x) for x in dbgris]
mGris=niv_gris_avg

#lissage de la moyenne (utiliser nombre impair) -> pas une bonne iddée
#niv_gris_avg_liss=myLissage(niv_gris_avg,3)
#mGris=niv_gris_avg_liss

#visu
plt.axis([0, 500, 0, 170])
plt.title('Niveaux de gris')
plt.plot(niv_gris_avg)
#plt.plot(niv_gris_avg_liss)
plt.show()

#histogramme
plt.hist(niv_gris_avg,200)  # 200 tranches
#plt.hist(niv_gris_avg_liss,200)
plt.title('Histograme du niveau de gris')
plt.axis([120, 160, 0, 250])
plt.show()

#calcul de la derivé (en pratique, c'est juste la difference)
derive_niv_gris_avg=[]
derive_niv_gris_avg.append(0) #fist element
for i in range(1,NbImages-1):
    derive_niv_gris_avg.append(niv_gris_avg[i]-niv_gris_avg[i-1])
derive_niv_gris_avg.append(0) #last element

#graphs
plt.axis([0, NbImages, -40, 40])
plt.title('Diff niveaux de gris')
plt.plot(derive_niv_gris_avg)
plt.show()

#histogramme
plt.hist(derive_niv_gris_avg,200)
plt.title('Histograme du diff niveau de gris')
plt.axis([-10, 10, 0, 80])
plt.show()

myResult=myCutOff(20,derive_niv_gris_avg)
precision, recall = myEvaluation(RepCode,myResult)
print(precision)
print(recall)

liste_precision = []
liste_recall = []
liste_myResult = []
liste_NbDetections = []

for i in range(1, 50):  # teste entre 1 et 50 en seuille
    liste_myResult = myCutOff(i, derive_niv_gris_avg)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)
    liste_NbDetections.append(len(liste_myResult))

# visu
plt.axis([0, len(liste_recall), 0, max(liste_precision) + 5])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

#visu
plt.axis([0, len(liste_precision), 0, max(liste_NbDetections)])
plt.title('Nombre de detections')
plt.plot(liste_NbDetections)
plt.show()

myResult=myCutOff(3,derive_niv_gris_avg)
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

#load
dbRGB = myLoadImage(RepImages,'listfull.txt','RGB')
dbRed=[]
dbBlue=[]
dbGreen=[]

for i in range(0,len(dbRGB)):
    (lBlue, lGreen, lRed) = cv2.split(dbRGB[i])
    dbBlue.append(lBlue)
    dbGreen.append(lGreen)
    dbRed.append(lRed)

#traitement 1 : la moyenne
niv_channel_avg =[np.mean(x) for x in dbRed]

#lissage de la moyenne (utiliser nombre impair)
#niv_channel_avg_liss=myLissage(niv_channel_avg,3)

mR=niv_channel_avg

#visu
plt.axis([0, 500, 0, 170])
plt.title('Niveaux de gris')
plt.plot(niv_channel_avg)
#plt.plot(niv_channel_avg_liss)
plt.show()

#histogramme
plt.hist(niv_channel_avg,200)  # 200 tranches
#plt.hist(niv_channel_avg_liss,200)
plt.title('Histograme du niveau de rouge')
plt.axis([110, 150, 0, 250])
plt.show()

#calcul de la derivé (en pratique, c'est juste la difference)
derive_channel=[]
derive_channel.append(0) #fist element
for i in range(1,len(niv_channel_avg)-1):
    derive_channel.append(niv_channel_avg[i]-niv_channel_avg[i-1])
derive_channel.append(0) #last element

#graphs
plt.axis([0, NbImages, -35, 35])
plt.title('Diff niveaux de rouge')
plt.plot(derive_channel)
plt.show()

#histogramme
plt.hist(derive_channel,200)
plt.title('Histograme du diff niveau de rouge')
plt.axis([-30, 30, 0, 100])
plt.show()

myResult=myCutOff(30,derive_channel)
precision, recall = myEvaluation(RepCode,myResult)
print(precision)
print(recall)

liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuille
    liste_myResult = myCutOff(i, derive_channel)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_recall), 0, max(liste_precision) + 5])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(3,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

# traitement 1 : la moyenne
niv_channel_avg = [np.mean(x) for x in dbGreen]

# lissage de la moyenne (utiliser nombre impair)
# niv_channel_avg_liss=myLissage(niv_channel_avg,3)
mG = niv_channel_avg

# calcul de la derivé (en pratique, c'est juste la difference)
derive_channel = []
derive_channel.append(0)  # fist element
for i in range(1, len(niv_channel_avg) - 1):
    derive_channel.append(niv_channel_avg[i] - niv_channel_avg[i - 1])
derive_channel.append(0)  # last element

# evaluation automatique
liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, derive_channel)
    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(2,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

# traitement 1 : la moyenne
niv_channel_avg = [np.mean(x) for x in dbBlue]

# lissage de la moyenne (utiliser nombre impair)
# niv_channel_avg_liss=myLissage(niv_channel_avg,3)
mB = niv_channel_avg

# calcul de la derivé (en pratique, c'est juste la difference)
derive_channel = []
derive_channel.append(0)  # fist element
for i in range(1, len(niv_channel_avg) - 1):
    derive_channel.append(niv_channel_avg[i] - niv_channel_avg[i - 1])
derive_channel.append(0)  # last element

# evaluation automatique
liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, derive_channel)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(3,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

#load
dbHSV = myLoadImage(RepImages,'listfull.txt','HSV')
dbH=[]
dbS=[]
dbV=[]

for i in range(0,len(dbHSV)):
    (lH, lS, lV) = cv2.split(dbHSV[i])
    dbH.append(lH)
    dbS.append(lS)
    dbV.append(lV)

#traitement 1 : la moyenne
niv_channel_avg =[np.mean(x) for x in dbH]

#lissage de la moyenne (utiliser nombre impair)
#niv_channel_avg_liss=myLissage(niv_channel_avg,3)
mH=niv_channel_avg

#visu
plt.axis([0, NbImages, 0, 200])
plt.title('Niveaux de H')
plt.plot(niv_channel_avg)
#plt.plot(niv_channel_avg_liss)
plt.show()

#histogramme à 200 tranches
plt.hist(niv_channel_avg,200)
plt.title('Histograme du niveau de H')
plt.axis([130, 180, 0, 250])
plt.show()

#calcul de la derivé (en pratique, c'est juste la difference)
derive_channel=[]
derive_channel.append(0) #fist element
for i in range(1,len(niv_channel_avg)-1):
    derive_channel.append(niv_channel_avg[i]-niv_channel_avg[i-1])
derive_channel.append(0) #last element

#graphs
plt.axis([0, NbImages, -35, 35])
plt.title('Diff niveaux de Hue')
plt.plot(derive_channel)
plt.show()

#histogramme
plt.hist(derive_channel,200)
plt.title('Histograme du diff niveau de Hue')
plt.axis([-20, 20, 0, 150])
plt.show()

myResult=myCutOff(30,derive_channel)
precision, recall = myEvaluation(RepCode,myResult)
print(precision)
print(recall)

liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, derive_channel)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(3,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

# traitement 1 : la moyenne
niv_channel_avg = [np.mean(x) for x in dbS]

# lissage de la moyenne (utiliser nombre impair)
# niv_channel_avg_liss=myLissage(niv_channel_avg,3)
mS = niv_channel_avg

# calcul de la derivé (en pratique, c'est juste la difference)
derive_channel = []
derive_channel.append(0)  # fist element
for i in range(1, len(niv_channel_avg) - 1):
    derive_channel.append(niv_channel_avg[i] - niv_channel_avg[i - 1])
derive_channel.append(0)  # last element

# evaluation automatique
liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, derive_channel)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(2,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

# traitement 1 : la moyenne
niv_channel_avg = [np.mean(x) for x in dbV]

# lissage de la moyenne (utiliser nombre impair)
# niv_channel_avg=myLissage(niv_channel_avg,3)
mV = niv_channel_avg

# calcul de la derivé (en pratique, c'est juste la difference)
derive_channel = []
derive_channel.append(0)  # fist element
for i in range(1, len(niv_channel_avg) - 1):
    derive_channel.append(niv_channel_avg[i] - niv_channel_avg[i - 1])
derive_channel.append(0)  # last element

# evaluation automatique
liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, derive_channel)

    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(3,derive_channel) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

#Calcul de la distance euclidienne
combi=[mR,mG,mB,mH,mS,mV,mGris]

liste_distance=[]
liste_distance.append(0) #initialisation
for i in range(1,NbImages):
    ldistance=0.0
    for channel in combi:
        ldistance+=(channel[i]-channel[i-1])**2
    liste_distance.append(math.sqrt(ldistance))

#graphs
plt.axis([0, NbImages, 0, 100])
plt.title('Diff distances avec 7 channels')
plt.plot(liste_distance)
plt.show()

#histogramme
plt.hist(liste_distance,200)
plt.title('Histograme du diff Diff distances avec 7 channels')
plt.axis([0, 100, 0, 100])
plt.show()

liste_myResult=myCutOff(3,liste_distance)
(lprecision, lrecall) = myEvaluation(RepCode,liste_myResult)
print(len(liste_myResult))

# evaluation automatique
liste_precision = []
liste_recall = []
liste_myResult = []

for i in range(1, 50):  # teste entre 1 et 50 en seuil
    liste_myResult = myCutOff(i, liste_distance)
    lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
    liste_precision.append(lprecision)
    liste_recall.append(lrecall)

# visu
plt.axis([0, len(liste_precision), 0,
          max(max(liste_precision), max(liste_recall))])
plt.title('Variation des indicateurs avec le cutoff')
plt.plot(liste_precision)
plt.plot(liste_recall)
plt.show()

# max
CutoffOptimal = liste_recall.index(max(liste_recall))
print('Best seuil :', CutoffOptimal)
print('Precision of best recall: ', liste_precision[CutoffOptimal])
print('Best Recall:', max(liste_recall))

myResult=myCutOff(8,liste_distance) #x+1
precision, recall = myEvaluation(RepCode,myResult)
print (precision, recall)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

nbBlocks=4 #pour 4x4
liste_im=myLoadImage(RepImages,'listfull.txt','gris')

liste_block_images=[]
for im in range(0,len(liste_im)):
    liste_block_images.append(myDivideBlocks(nbBlocks,liste_im[im]))

#Verification par example avec l'image 200
plt.figure()
for i in range(0,nbBlocks**2):
    plt.subplot(4,4,i+1)
    plt.imshow(liste_block_images[200][i],cmap='gray')

#calcul de la moyenne de chaque block suivi de lisage

#le lissage n'est peut etre pas une bonne idée vu le faible grand pas de temps de l'extraction

megaliste=[]
for iblock in range(0,nbBlocks**2):
    liste=[]
    #liste_moyenne_block_lisse_par_image=[]
    for im in range(0,NbImages):
        liste.append(np.mean(liste_block_images[im][iblock][:,:,]))
        #liste_moyenne_block_lisse_par_image=myLissage(liste,3)
    #megaliste.append(liste_moyenne_block_lisse_par_image)
    megaliste.append(liste)

#visu
plt.figure()
for i in range(0,nbBlocks**2):
    plt.subplot(4,4,i+1)
    plt.xticks([]), plt.yticks([])
    plt.axis([0, NbImages, 0, 250])
    plt.plot(megaliste[i])

#histogrammes
plt.figure()
for i in range(0,nbBlocks**2):
    plt.subplot(4,4,i+1)
    plt.hist(megaliste[i],200)  # 200 tranches
    plt.xticks([]), plt.yticks([])
    plt.axis([0, 250, 0, 100])

#calcul de la derivé (en pratique, c'est juste la difference)
derive_par_block=[]
for iblock in range(0,nbBlocks**2):
    derive=[]
    derive.append(0)
    for im in range(1,NbImages):
        derive.append(megaliste[iblock][im]-megaliste[iblock][im-1])
    derive_par_block.append(derive)

#graphs
plt.figure()
for i in range(0,nbBlocks**2):
    plt.subplot(4,4,i+1)
    plt.xticks([]), plt.yticks([])
    plt.axis([0, NbImages, -150, 150])
    plt.plot(derive_par_block[i])

#histogramme
plt.figure()
for i in range(0,nbBlocks**2):
    plt.subplot(4,4,i+1)
    plt.xticks([]), plt.yticks([])
    plt.hist(derive_par_block[i],200)
    plt.axis([-20, 20, 0, 80])

seuil1 = 10
seuil2 = 0.4
myResult = []

for im in range(0, len(liste_im)):
    blockcounteur = 0
    for iblock in range(0, nbBlocks ** 2):
        if abs(derive_par_block[iblock][im]) > seuil1:
            blockcounteur += +1
    if blockcounteur / float(nbBlocks ** 2) >= seuil2:
        myResult.append(im)

precision, recall = myEvaluation(RepCode, myResult)
print(precision)
print(recall)

liste_liste_precision = []
liste_liste_recall = []

for seuil1 in range(0, 100, 10):
    liste_precision = []
    liste_recall = []

    for seuil2 in range(0, 100, 10):
        liste_myResult = []

        for im in range(0, NbImages):
            blockcounteur = 0
            for iblock in range(0, nbBlocks ** 2):
                if abs(derive_par_block[iblock][im]) > seuil1:
                    blockcounteur = blockcounteur + 1

            if blockcounteur / float(nbBlocks ** 2) >= seuil2 / float(100):
                liste_myResult.append(im)

        lprecision, lrecall = myEvaluation(RepCode, liste_myResult)
        liste_precision.append(lprecision)
        liste_recall.append(lrecall)

    liste_liste_precision.append(liste_precision)
    liste_liste_recall.append(liste_recall)

# visu
#  y est le recall
#  x est le seuil2 sur le nombre le % blockes (ex: 5 => 50% => 8 sur 16)
# chaque courbe correspond au seuil1 (niveau de modification sur 1 block)
for i in range(0, 10):
    plt.plot(liste_liste_recall[i], label=(i) * 10)
    plt.legend(ncol=1, borderaxespad=0.)

#visu
for i in range(0,10):
    plt.plot(liste_liste_precision[i],label=i)
    plt.legend(ncol=1, borderaxespad=0.)

myWriteCSV(RepCode+'myResult.csv',myResult)  #ne pas changer le nom

