# -*- coding: utf-8 -*-
import os
import cv2

def extractFrames(pathIn, pathOut,fps):
    #os.mkdir(pathOut)
    
    f = open(pathOut+'list.txt', 'w+')
    cap = cv2.VideoCapture(pathIn)
    #cap.set(cv2.CAP_PROP_FPS, fps)

    count = 0
    i=0
 
    while (cap.isOpened()):
 
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if i==0 or i%fps==0:
                print('Read %d frame: ' % count, ret, ' ', int(float(count)/(fps*60)))
                cv2.imwrite(os.path.join(pathOut, "{:08d}.jpg".format(count)), frame)  # save frame as JPEG file
                f.write("{:08d}.jpg\n".format(count))
                count += 1
            i+=1
        else:
            break

    f.close() 
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
Out = r"C:\Users\costav\Documents\big data\Ecole\Multimedia\data5FPS\\"
In=r"C:\Users\costav\Documents\big data\Ecole\Multimedia\\"

extractFrames(In+'06-11-22.mp4', Out,5)
 