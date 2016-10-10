import imutils as imutils
from skimage.exposure import exposure
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np

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
    return i,NOL,NumOfLines,img,separate_line

def separate_outside(img_to_gray,img_be_draw,fact_submax,fact_length,precise):
    gray = cv2.cvtColor(img_to_gray, cv2.COLOR_RGB2GRAY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    row, col = gray.shape
    for n in range(100):
        i,NOL,NumOfLines,ret_img,separate_line=separate_inside(gray.copy(),img_be_draw,fact_submax,fact_length,row,col)
        if i<(row-1) and NOL==NumOfLines:
            fact_submax-precise
            continue
        if i==row-1 and NOL<NumOfLines:
            fact_submax+precise
            continue
        if i==row-1 and NOL==NumOfLines:
            break
    cv2.imwrite("ret_img.png",ret_img )
    return separate_line

def optimize_separate(img_to_find,separate_line):
    sum=0
    for i in range(len(separate_line)-2):
        sum+=separate_line[i+1]-separate_line[i]
    average=sum/len(separate_line)
    for i in range(len(separate_line)-2):
        sub=separate_line[i+1]-separate_line[i]
        if sub<average*0.5:
            del separate_line[i+1]
    return separate_line



#def find_num_img(img_to_find,separate_line):



digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
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
number=[]*5
precise=0.01
fact_submax=0.6
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


#separate numbers in the captched image

'''1.Need to add code accociated with image rotation'''

'''2.Must pass the image that in colorful format'''

separate_line=separate_outside(warp,warp,fact_submax,fact_length,precise)
separate_line=optimize_separate(warp,separate_line)

cv2.waitKey(0)


#raw_input()
# cv2.imwrite( "d:/123.png", colorCVT );


