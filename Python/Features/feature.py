
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
            sharp = imageSharpness(imageBox(image,t,l,w,h),w,h)
            lum   = imageIllumination(imageBox(image,t,l,w,h))
            #pose=HOGpose(image,t,l,w,h,predictor,detector)
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

def imageSharpness(image,w,h):
    ratio = float(90000) /float(image.shape[0] * image.shape[1])
    image =cv2.resize(image, (300, 300), fx=ratio, fy=ratio)
    image1=image[50:250,50:250]
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray,cv2.CV_64F).var()
    #fm1 = cv2.Laplacian(gray1,cv2.CV_64F).var()
    return fm

''' word gebruik om face box te maak waarop die sharpness en Illumination gedoen word '''
def imageBox(img,t,l,w,h):
    crop_img = img[t:t+h,l:l+w]
    #cv2.imshow("Crop Image: ",crop_img)
    #key = cv2.waitKey(0)
    return crop_img

def imageIllumination(image):
    ratio = float(90000) /float(image.shape[0] * image.shape[1])
    image =cv2.resize(image, (300, 300), fx=ratio, fy=ratio)
    image1=image[50:250,50:250]
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray = gray.ravel()
    gray =np.array(gray)
    gray.sort()
    len=gray.shape[0]
    cutPer=int(len*0.05)
    newGray=gray[cutPer:len-cutPer]
    band=newGray.max()-newGray.min()
    lum=band/266
    return lum

def HOGpose(img,t,l,w,h,predictor,detector):
    a=[l,t,l+w,t+h]
    dets = detector(img, 1)
    for i, d in enumerate(dets):
        b=[ d.left(), d.top(), d.right(), d.bottom()]
        area=rectIntersect(a,b)
        if(area > 0.5):
            out_face = np.zeros_like(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape = predictor(gray, d)
            shape = face_utils.shape_to_np(shape)
            dist = np.sqrt( (shape[28][0] - shape[40][0])**2 + (shape[28][1] - shape[40][1])**2 )
            dist1 = np.sqrt( (shape[43][0] - shape[28][0])**2 + (shape[43][1] - shape[28][1])**2 )
            fdist=min(dist,dist1)


        else:
            fdist=0

    if(len(dets)==0):
        fdist=0
    return fdist
''' hog gebruik die function om te kyk of sy box met mmod overlap '''
def rectIntersect(a,b):
    left = max(a[0], b[0])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    top = max(a[1], b[1])

    if((left < right) and (bottom > top)):
        intArea=(right-left)*(bottom-top)
        mmodArea=(a[2]-a[0])*(a[3]-a[1])

        return intArea/mmodArea
    else:
        return 0

def mmodpose(img,t,l,w,h,predictor):
    val=0
    h,w,c=img.shape
    top=t
    left=l
    right=l+w
    bottom=t+h
    if(0> top):
        top=0
    if(0> left):
        left=0
    if(w < right):
        right=w
    if(h< bottom):
        bottom=h
    out_face = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rect=dlib.rectangle(left,top,right,bottom)
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    m1=  -1*(shape[2][1] - shape[0][1])/(shape[2][0] - shape[0][0])

    c1=shape[2][1]-shape[2][0]*m1
    if(m1==0):
        X=shape[4][0]
        Y=shape[2][1]
    else:
        m2=1/m1
        c2=shape[4][1]-shape[4][0]*m2
        X=(c2-c1)/(m1-m2)
        Y=m1*X+c1
    d1 = np.sqrt( (shape[4][0] - X)**2 + (shape[4][1] - Y)**2 )
    d2 = np.sqrt( (shape[0][0] - X)**2 + (shape[0][1] - Y)**2 )
    long =max(d1,d2)
    short= min(d1,d2)
    val =min(d1,d2)/((d1+d2)/2)
    val=val*0.5+0.5
    eyeDistance= np.sqrt( (shape[2][0] - shape[0][0])**2 + (shape[2][1] - shape[0][1])**2 )
    if (eyeDistance>d1+d2):
        val = 1-short/long
        val = val*0.5
    print(val)
    return val










readCSV(path)
