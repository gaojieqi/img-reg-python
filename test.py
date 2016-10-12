import imutils as imutils
import sys

import sklearn
from skimage.exposure import exposure
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.datasets import mnist


def compare(list_element,element):
    flag=0
    for i in range(len(list_element)):
        if list_element[i]==element:
            flag=1
    return flag

#define outer and inner funtion to separate the numbers
def separate_inside(gray,img,fact_submax,fact_length,row,col):
    submax = 0
    end_flag=0
    NOL=0
    NumOfLines=1
    buffer=[0]
    dif=0
    separate_line=[0]
    for i in range(col - 1):
        if end_flag==1:
            i-=1
            break
        subsum = 0
        for j in range(row - 1):
            subsum += gray[j][i]
        if submax < subsum:
            submax = subsum
        if subsum < fact_submax * submax:
            buffer[NOL] = i
            if NOL != 0:
                dif=buffer[NOL]-buffer[NOL-1]
                if dif < temp*fact_length:
                    continue
            temp=dif
            submax = 0
            for j in range(row - 1):
                img[j][i] =0
            last_line_col=i
            separate_line[NOL]=i
            separate_line.append(0)
            if col-last_line_col>0.1*col:
                NumOfLines+=1
                buffer.append([0])
            NOL+=1
            if NOL==NumOfLines:
                end_flag=1
                break
    del separate_line[-1]
    return i,NOL,NumOfLines,img,separate_line

def separate_outside(img_in_gray,fact_submax,fact_length,precise):
    row, col = img_in_gray.shape
    for n in range(100):
        i,NOL,NumOfLines,ret_img,separate_line=separate_inside(gray,gray.copy(),fact_submax,fact_length,row,col)
        if i<(row-1) and NOL==NumOfLines:
            fact_submax-precise
            continue
        if i==row-1 and NOL<NumOfLines:
            fact_submax+precise
            continue
        if i==row-1 and NOL==NumOfLines:
            break
    cv2.imwrite("ret_img.png",ret_img )
    return separate_line,ret_img

def optimize_separate(separate_line,img_to_test):
    sum=separate_line[-1]-separate_line[0]
    average=sum/(len(separate_line)-1)
    buff=[0]
    num=0
    row,col,dimension = img_to_test.shape
    for i in range(len(separate_line)-2):
        sub=separate_line[i+1]-separate_line[i]
        print sub,average
        if sub<average*0.8:
            separate_line[i]=0
    for i in range(len(separate_line)-1):
        if separate_line[i]!=0:
            buff[num]=separate_line[i]
            num+=1
            buff.append(0)
    for i in range(len(buff)-1):
        for j in range(row - 1):
            img_to_test[j][buff[i]] = 0
    cv2.imwrite('11111111.jpg',img_to_test)
    del buff[-1]
    return buff

def find_num_img(img_to_find,separate_line,quantity):
    row, col = gray.shape
    SUBSUM=[0]
    list1=[0]*quantity
    for i in range(len(separate_line)-2):
        subsum=0
        sub = separate_line[i + 1] - separate_line[i]
        for j in range(separate_line[i],separate_line[i+1]):
            for k in range(row-1):
                subsum+=gray[k][j]
        SUBSUM[i]=subsum
        SUBSUM.append(0)
    del SUBSUM[-1]
    SUBSUM=sorted(SUBSUM,reverse=True)
    list1[0:quantity]=SUBSUM[0:quantity]
    for i in range(len(separate_line)-2):
        subsum=0
        sub = separate_line[i + 1] - separate_line[i]
        for j in range(separate_line[i],separate_line[i+1]):
            for k in range(row-1):
                subsum+=gray[k][j]
        if compare(list1,subsum)==0:
            for j in range(separate_line[i], separate_line[i + 1]):
                for k in range(row - 1):
                    gray[k][j]=255
    cv2.imwrite("gray_return.png", gray)


'''
1.while a high C aims at classifying all training examples correctly
2.The larger gamma is, the closer other examples must be to be affected.
'''
img=cv2.imread('test2.jpg')
#Filt out some noise
img = cv2.bilateralFilter(img, 11, 17, 17)
#detect the boarder
edge=cv2.Canny(img,10,200)
ratio=img.shape[0]/300.0
row,col,demension=img.shape
quantity=5
precise=0.1
fact_submax=0.8
fact_length=0.8
#


#take fourier transfer action on img
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
fourier=np.fft.fft2(gray)
f=np.fft.fftshift(fourier)
temp = 20*np.log(np.abs(f))
cv2.imwrite('fourier.jpg',temp)
img1=cv2.imread('fourier,jpg')
retval,f_detect=cv2.threshold(img1,100,150,cv2.THRESH_BINARY)
#retval,f_detect=cv2.adaptiveThreshold(img1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
edge_fourier=cv2.Canny(img1,10,50)
cv2.imwrite('edge_fourier.jpg',edge_fourier)
lines= cv2.HoughLines(edge_fourier,1,np.pi/180,100)

#find the contours and regonize the rectangle
image_find_contour,contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
screenCnt=None
for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    if len(approx)==4:
        screenCnt=approx
        break
cv2.drawContours(img, [screenCnt], -1, (0, 255, 127), 3)
cv2.imwrite('rectangle.jpg',img)


#find the points of rectangle
pts=screenCnt.reshape(4,2)
rect=np.zeros((4,2),dtype="float32")
s=pts.sum(axis=1)
rect[0]=pts[np.argmin(s)]
rect[2]=pts[np.argmax(s)]
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]

#compute the shape of new rectangle
(tl, tr, br, bl) = rect
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(img, M, (maxWidth,maxHeight))
cv2.imwrite("warp.png",warp)


#find numbers in KNearest

'''1.Need to add code accociated with image rotation'''

'''2.Must pass the image that in colorful format'''

# gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
# ret,gray = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)
# cv2.imwrite('gray.jpg',gray)
# img_find_contour,contours,hierarchy= cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# samples =  np.empty((0,100))
# responses = []
# keys = [i for i in range(48,58)]
# num_buff=0
# for cnt in contours:
#     if cv2.contourArea(cnt)>0.01*gray.size and cv2.contourArea(cnt)<warp.size*0.05:
#         rect = cv2.minAreaRect(contours[0])
#         [x, y, w, h] = cv2.boundingRect(cnt)
#         if h > 28:
#             cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 0, 255), 2)
#             roi = gray[y:y + h, x:x + w]
#             roismall = cv2.resize(roi, (10, 10))
#             cv2.imshow('norm', warp)
#             key = cv2.waitKey(0)
#             if key == 27:
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1, 100))
#                 samples = np.append(samples, sample, 0)
#                 num_buff+=1
# responses = np.array(responses,np.float32)
# responses = responses.reshape((responses.size,1))
# print "training complete"
# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)


#find numbers in SVM
digits=datasets.load_digits()
clf = svm.SVC(gamma=0.001,C=100)
n_samples = len(digits.images)
x,y=digits.data[:-10],digits.target[:-10]
clf.fit(x,y)
predict = np.array(digits.data[-6]).reshape((1, -1))

vectorizer=TfidfVectorizer()
vectors= vectorizer.fit_transform()
x_train, y_train =datasets.load_svmlight_file("optdigits.tra")
x_test, y_test = datasets.load_svmlight_files("optdigits.tes")

(X_train, y_train), (X_test, y_test) = mnist.load_data()
clf1=svm.SVC(gamma=0.001,C=100)
#clf1.fit(X_train,y_train)
print X_train.shape,y_train.shape

gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
ret,gray = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
cv2.imwrite('gray.jpg',gray)
img_find_contour,contours,hierarchy= cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]
num_buff=0
for cnt in contours:
    if cv2.contourArea(cnt)>0.01*gray.size and cv2.contourArea(cnt)<warp.size*0.05:
        rect = cv2.minAreaRect(contours[0])
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h > 28:
            cv2.rectangle(warp, (x, y), (x + w, y + h), (0, 0, 255), 2)
            roi = gray[y:y + h, x:x + w]
            roismall = cv2.resize(roi, (8, 8))
            cv2.imshow('norm', warp)
            sample = roismall.reshape((1, 64))
            print clf1.predict(sample)
            key = cv2.waitKey(0)
            if key == 27:
                 sys.exit()

#raw_input()
# cv2.imwrite( "d:/123.png", colorCVT );


