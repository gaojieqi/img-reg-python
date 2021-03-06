import sys
from sklearn import svm
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.datasets import mnist
from sklearn.externals import joblib
from sklearn import datasets
from matplotlib import pyplot as plt
import DetectChars
import DetectPlates
import PossibleChar
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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

def neural_network_model(data):
    hidden_1_layer={'weights':tf.Variable(tf.random_normal([feature_num,n_nodes_hl1])),
                    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) , hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']) , hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3, output_layer['weights']) , output_layer['biases'])
    return output

















img=cv2.imread('test1.jpg')
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


###############################################find the contours and regonize the rectangle
image_find_contour,contours,hierarchy = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(contours,key=cv2.contourArea,reverse=True)[:10]
screenCnt=None
for c in cnts:
    peri=cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,0.02*peri,True)
    cv2.drawContours(img, [approx], -1, (0, 255, 127), 3)
    if len(approx)==4:
        screenCnt=approx
        break
cv2.drawContours(img, [screenCnt], -1, (255, 0, 255), 3)
cv2.imwrite('rectangle.jpg',img)


#############################################find the points of rectangle
pts=screenCnt.reshape(4,2)
rect=np.zeros((4,2),dtype="float32")
s=pts.sum(axis=1)
rect[0]=pts[np.argmin(s)]
rect[2]=pts[np.argmax(s)]
diff = np.diff(pts, axis = 1)
rect[1] = pts[np.argmin(diff)]
rect[3] = pts[np.argmax(diff)]


############################################compute the shape of new rectangle
(tl, tr, br, bl) = rect

widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))
maxHeight = max(int(heightA), int(heightB))

if maxHeight>maxWidth:
    temp=maxHeight
    maxHeight=maxWidth
    maxWidth=temp
    rect=np.array([tr,br,bl,tl],dtype="float32")

dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(img, M, (maxWidth,maxHeight))
warp = cv2.bilateralFilter(warp, 11, 17, 17)
gray = cv2.cvtColor(warp, cv2.COLOR_RGB2GRAY)
cv2.imwrite("warp.png",warp)


################################################find numbers in KNearest

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
# responses = responses.reshape((responses.size, 1))
# print "training complete"
# np.savetxt('generalsamples.data',samples)
# np.savetxt('generalresponses.data',responses)




#####################################################find numbers

'''
1.while a high C aims at classifying all training examples correctly
2.The larger gamma is, the closer other examples must be to be affected.
'''



'''Random Forest Algorithm'''
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train=X_train.reshape(60000,784)
# clf=RandomForestClassifier(n_estimators=2000)
# clf.fit(X_train,y_train)
# joblib.dump(clf, "train_model.m")
# clf = joblib.load("train_model.m")
# print clf.score(X_test.reshape(10000,784),y_test)


'''SVM'''
# digits = datasets.load_digits()
# (X_train, y_train)=digits.data[:-1],digits.target[:-1]
# clf=svm.SVC(gamma=1,C=1,probability=True)
# clf.fit(X_train,y_train)


# clf.fit(X_train,y_train)
# joblib.dump(clf, "svm_model.m")
# clf = joblib.load("svm_model.m")
# print clf.score(X_test.reshape(10000,784),y_test)



'''kNearest'''
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train=X_train.reshape(60000,784).astype(np.float32)
# y_train=y_train.astype(np.float32)
# kNearest = cv2.ml.KNearest_create()
# kNearest.train(X_train,cv2.ml.ROW_SAMPLE,y_train)


'''Neural Network'''
mnist= input_data.read_data_sets("/tmp/data/",one_hot=True)
sess=tf.InteractiveSession()
n_nodes_hl1=500
n_nodes_hl2=500
n_nodes_hl3=500
n_classes=10
batch_size=100
feature_num=784
X=tf.placeholder('float',[None,feature_num])
Y=tf.placeholder('float')
prediction=neural_network_model(X)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,Y))
optimizer=tf.train.AdamOptimizer().minimize(cost)
hm_epochs=30
#-------------------save model-------------
# sess.run(tf.initialize_all_variables())
# for epoch in range(hm_epochs):
#     epoch_loss=0
#     for _ in range(int(mnist.train.num_examples/batch_size)):
#         epoch_x,epoch_y=mnist.train.next_batch(batch_size)
#         _,c=sess.run([optimizer,cost],feed_dict={X:epoch_x,Y:epoch_y})
#         epoch_loss+=c
#     print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)
# saver=tf.train.Saver()
# saver.save(sess,'Neural_Network.model')

#-------------------load model-------------
saver=tf.train.Saver()
saver.restore(sess,'Neural_Network.model')









'''use testimg to test the accuracy of the programme ***factor_h needs to be 0.3 ***'''
img = cv2.imread('testimg.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11, 2)
cv2.imwrite('thresh.jpg',thresh)


img_find_contour,contours,hierarchy= cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
out = np.zeros(img.shape,np.uint8)
H,W=gray.shape


'''factor that makes the digits to the center of the image'''
factor_h=0.3
factor_w=0.2

'''minimal h of digits's hight'''
H_MIN=28

'''Flag that distinct digits from aphalet'''
FLAG=100

blnKNNTrainingSuccessful = DetectChars.loadKNNDataAndTrainKNN()         # attempt KNN training
listOfPossibleChars = []


for cnt in contours:
        [x,y,w,h] = cv2.boundingRect(cnt)
        # if (w>0.1*W) and (w<0.3*W) and (h>H*0.5) :            #use in ordinary circumstance
        if h>H_MIN:                                             #use in testing
            cv2.rectangle(warp,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.imwrite('warp.png', warp)
            num = thresh[(y-int(factor_h*h)):(y+h+int(h*factor_h)),(x-int(factor_w*w)):(x+w+int(w*factor_w))]
            # num= thresh[y:(y + h ), x :(x + w)]
            H1,W1=num.shape
            for i in range(W1):
                for j in range(H1):
                    if i not in range(int(factor_w*w),w+int(factor_w*w)) or j not in range(int(factor_h*h),int(factor_h*h)+h):
                        num[j,i]=0
            plt.imshow(num, 'gray')
            plt.show()

            num_possible = PossibleChar.PossibleChar(num)
            # listOfPossibleChars.append(possibleChar)
            # listOfListsOfMatchingCharsInPlate=DetectChars.findListOfListsOfMatchingChars(listOfPossibleChars)
            #
            # for i in range(0, len(listOfListsOfMatchingCharsInPlate)):  # within each list of matching chars
            #     listOfListsOfMatchingCharsInPlate[i].sort(key=lambda matchingChar: matchingChar.intCenterX)  # sort chars from left to right
            #     listOfListsOfMatchingCharsInPlate[i] = DetectChars.removeInnerOverlappingChars( listOfListsOfMatchingCharsInPlate[i])  # and remove inner overlapping chars
            # intLenOfLongestListOfChars = 0
            # intIndexOfLongestListOfChars = 0
            #
            # # loop through all the vectors of matching chars, get the index of the one with the most chars
            # for i in range(0, len(listOfListsOfMatchingCharsInPlate)):
            #     if len(listOfListsOfMatchingCharsInPlate[i]) > intLenOfLongestListOfChars:
            #         intLenOfLongestListOfChars = len(listOfListsOfMatchingCharsInPlate[i])
            #         intIndexOfLongestListOfChars = i
            #         # end if
            #         # end for
            # longestListOfMatchingCharsInPlate = listOfListsOfMatchingCharsInPlate[intIndexOfLongestListOfChars]

            if num_possible.fltAspectRatio != 0:
                num_possible=DetectChars.removeInnerOverlappingChars(num_possible)
                num_detect = DetectChars.recognizeCharsInPlate(num, num_possible)

                # ---------------------------dataset 1(Unknown) ---------------------------------
                # if num_detect!=FLAG:
                #     cv2.putText(out, str(num_detect), (x, y + h), 0, 1, (0, 255, 0))
                #     print num_detect
                #     cv2.imwrite('out.png', out)
                #
                # # ---------------------------dataset 2(60000)  Knearest ---------------------------------
                # else:
                #     numsmall = cv2.resize(num, (28, 28))
                #     ret, numsmall = cv2.threshold(numsmall, 1, 255, cv2.THRESH_BINARY)
                #     numsmall = numsmall.reshape((1, 784))
                #     numsmall = np.float32(numsmall)
                #     retval, npaResults, neigh_resp, dists = kNearest.findNearest(numsmall, k=1)
                #     cv2.putText(out, str(int(npaResults[0][0])), (x, y + h), 0, 1, (0, 255, 0))
                #     print int(npaResults[0][0])
                #     cv2.imwrite('out.png', out)

                # ---------------------------dataset 2(60000)  SVM ---------------------------------
                # else:
                #     numsmall = cv2.resize(num, (28, 28))
                #     ret, numsmall = cv2.threshold(numsmall, 1, 255, cv2.THRESH_BINARY)
                #     numsmall = numsmall.reshape((1, 784))
                #     result=clf.predict(numsmall)
                #     cv2.putText(out, str(result), (x, y + h), 0, 1, (0, 255, 0))
                #     cv2.imwrite('out.png', out)


                # ---------------------------dataset 3(55000)   Nueral Network ---------------------------------
                numsmall = cv2.resize(num, (28, 28))
                ret, numsmall = cv2.threshold(numsmall, 1, 255, cv2.THRESH_BINARY)
                numsmall = numsmall.reshape(1, 784)
                numsmall = np.float32(numsmall)
                pre = tf.argmax(prediction, 1)
                print sess.run(pre, {X: numsmall})
                # cv2.putText(out, str(), (x, y + h), 0, 1, (0, 255, 0))
                # cv2.imwrite('out.png', out)

