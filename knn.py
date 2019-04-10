import sklearn
import numpy
import math
import os
import matplotlib as plt
import cv2


def myLoadImage(aFullPath, aFileName, aTypeColor='gris', abreak=None):
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
    # lecture
    GoldList = []
    with open(aRep + 'goldResult.csv', encoding="utf-8") as f:
        for line in f:
            GoldList.append(int(line) - 1)
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


def myLissage(aList, aNbImagesLissage=5):
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
    lResult = []
    for i in range(len(aList)):
        if abs(aList[i]) > aCutOff:
            lResult.append(i)
    return lResult


def myDivideBlocks(aNbBlocks, aImage):
    (dimYpix, dimXpix) = aImage.shape

    dimYpixBlock = int(dimYpix / aNbBlocks)
    dimXpixBlock = int(dimXpix / aNbBlocks)

    liste_block_frame = []
    for bl in range(0, aNbBlocks ** 2): liste_block_frame.append(aImage
                                                                [(bl // aNbBlocks) * dimYpixBlock: (
                                                                                                              bl // aNbBlocks + 1) * dimYpixBlock,
                                                                (bl % aNbBlocks) * dimXpixBlock:(
                                                                                                           bl % aNbBlocks + 1) * dimXpixBlock])

    return liste_block_frame


path = r"E:\Dropbox\CES DAta sciences\Cours\Module 4\Projet\data\\"
RepCode = r"E:\Dropbox\CES DAta sciences\Cours\Module 4\Projet\evasion\\"

os.chdir(path)
RepImages = (path)

nbBlocks = 4  # pour 4x4
liste_im = myLoadImage(RepImages, 'listfull.txt', 'gris')

NbImages = len(liste_im)

liste_block_images = []
for im in range(NbImages):
    liste_block_images.append(myDivideBlocks(nbBlocks, liste_im[im]))

lhistblock = []
for iblock in range(nbBlocks ** 2):
    liste = []
    for im in range(NbImages):
        block_hist = cv2.calcHist(liste_block_images[im][iblock], [0], None, [64], [0, 255])
        liste.append(cv2.normalize(block_hist, block_hist, 1.0, 0.0, cv2.NORM_L1))
    lhistblock.append(liste)

for i in liste_im:
    print(i)