from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import cv2

digits=datasets.load_digits()
clf=svm.SVC(gamma=0.001,C=100)
'''
1.while a high C aims at classifying all training examples correctly
2.The larger gamma is, the closer other examples must be to be affected.
'''
print len(digits.data)

x,y=digits.data[:-1],digits.target[:-1]
clf.fit(x,y)
clf.predict()
plt.imshow(digits.images[-2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()


#raw_input()

# cv2.imwrite( "d:/123.png", colorCVT );


