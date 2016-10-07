from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2
import numpy as np

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
'''
1.while a high C aims at classifying all training examples correctly
2.The larger gamma is, the closer other examples must be to be affected.
'''
img=cv2.imread('abc.jpg',0)
edges=cv2.Canny(img,100,200)
fourier=np.fft.fft2(img)
lines= cv2.HoughLines(edges,1,np.pi/180,200)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)






#raw_input()

# cv2.imwrite( "d:/123.png", colorCVT );


