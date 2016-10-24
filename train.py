import numpy as np
import cv2
from matplotlib import pyplot as plt
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(60000,784).astype(np.float32)
y_train=y_train.astype(np.float32)
model = cv2.ml.KNearest_create()
model.train(X_train,cv2.ml.ROW_SAMPLE,y_train)

np.savetxt('samples.txt',X_train)
np.savetxt('classes.txt',y_train)
