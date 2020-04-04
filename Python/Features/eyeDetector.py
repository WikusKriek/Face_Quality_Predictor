
from imutils import face_utils
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import collections
import dlib
import csv
import pandas as pd
eye_cascade = cv2.CascadeClassifier('/root/opencv/data/haarcascades/haarcascade_eye.xml')
data=''

'''
Die program gebruik die labeled data wat ons gestoor het in die csv file met die node program.
Hy gee vir ons al die features van die image box wat hy in die csv file snap_scores.csv kry.
Die features word dan in n nuwe csv file gestoor wat dan deur die Decision_forrest.py program gebruik word om die model op te stel.

snapFolder id die path na die snaps toe
path is die path na die csv file waaruit ons die face box, confidence, image id en al daai goed kry
'''
snapFolder='/home/wmk/IQA/snaps/'
path ="/home/wmk/IQA/Labled_data/snap_scores.csv"


def readCSV (path):
    string='SnapId'+','+'Confidence'+','+'Sharpness'+','+'Size'+','+'Illumination'+','+'Pose'+','+'Good'+'\n'
    file=open( path, "r")
    reader = pd.read_csv(file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_5_face_landmarks.dat")
    for i,row in reader.iterrows():
        ''' daar is baie rows in die csv file wat die node program gegee het wat nan values het of undefined is so die if was net om te kyk of is die foto al gelable
        '''
        if((row.UserEstimate=='1')or(row.UserEstimate=='2')or(row.UserEstimate=='3')or(row.UserEstimate=='4')or(row.UserEstimate=='5')):
            snapPath=snapFolder+row.SnapId+'.jpeg'
            image = cv2.imread(snapPath)

            ''' maak seker die snap gaan nie oor die borders van die foto nie'''
            t=max(int(row.Top), 0)
            l=max(int(row.Left), 0)
            w=min(int(row.Right), image.shape[1])-l
            h=min(int(row.Bottom), image.shape[0])-t

            ''' kry al die features'''
            confidence=row.Confidence
            size  = w*h
            eyeDetector(imageBox(image,t,l,w,h))

            '''vir nou los ons hog pose uit dit vat anyway 10 keer die tyd'''
            pose=mmodpose(image,t,l,w,h,predictor)
            if(int(row.UserEstimate) >=3):
                good='True'
            else:
                good='False'

            '''save die features in die string en add elke keer new line ..... stoor die string dan as Data.csv '''
            string=string+row.SnapId+','+str(confidence)+','+str(sharp)+','+str(size)+','+str(lum)+','+str(pose)+','+good+'\n'
            f = open('Data1.csv','w')
            f.write(string)
            f.close()
    return



''' word gebruik om face box te maak waarop die sharpness en Illumination gedoen word '''
def imageBox(img,t,l,w,h):
    crop_img = img[t:t+h,l:l+w]
    #cv2.imshow("Crop Image: ",crop_img)
    #key = cv2.waitKey(0)
    return crop_img



def eyeDetector(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


    eyes = eye_cascade.detectMultiScale(gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()









readCSV(path)
