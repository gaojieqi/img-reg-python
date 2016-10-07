import imutils as imutils
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np
form skimage import rescale_intensity

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
'''
1.while a high C aims at classifying all training examples correctly
2.The larger gamma is, the closer other examples must be to be affected.
'''


img=cv2.imread('test1.jpg')
img = cv2.bilateralFilter(img, 11, 17, 17)#Filt out some noise
edge=cv2.Canny(img,10,200)#detect the boarder
ratio=img.shape[0]/300.0

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
image,contours,hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

#find the point of rectangle
pts=screenCnt.reshape(4,2)
rect=np.zeros((4,2),dtype="float32")
s=pts.sum(axis=1)
rect[0]=pts[np.argmin(s)]
rect[2]=pts[np.argmax(s)]
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]
rect *= ratio

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
warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
warp = exposure.rescale_intensity(warp, out_range=(0, 255))

# the pokemon we want to identify will be in the top-right
# corner of the warped image -- let's crop this region out
(h, w) = warp.shape
(dX, dY) = (int(w * 0.4), int(h * 0.45))
crop = warp[10:dY, w - dX:w - 10]

# save the cropped image to file
cv2.imwrite("cropped.png", crop)

# show our images
cv2.imshow("image", image)
cv2.imshow("edge", edge)
cv2.imshow("warp", imutils.resize(warp, height=300))
cv2.imshow("crop", imutils.resize(crop, height=300))
cv2.waitKey(0)


#raw_input()
# cv2.imwrite( "d:/123.png", colorCVT );


