import numpy as np 
import pandas as pd
from keras.datasets import cifar10
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm
from sklearn.metrics import accuracy_score
from PIL import Image
from numpy import asarray
import numpy as np
import math
import cv2
import pickle

(x_Train, yTrain), (x_Test, yTest) = cifar10.load_data()
x_train = x_Train[:50000]
y_train = yTrain[:50000]

x_test = x_Test[:10000]
y_test = yTest[:10000]

x_train = x_train.astype('float64')
x_train /= 255

x_test = x_test.astype('float64')
x_test /= 255

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

x_train_hog = []
for i in range(len(x_train)):
    fd , hog_im = hog(x_train[i] , orientations=9 , pixels_per_cell = (8,8),
        cells_per_block = (2,2) , visualize = True ,  multichannel = True)
    x_train_hog.append(fd)

x_train_hog = np.array(x_train_hog)

my_model = svm.SVC(kernel = 'rbf')
my_model.fit(x_train_hog , y_train.ravel())

pickle.dump(my_model,open('model.pkl','wb'))

x_test_hog = []
for i in range(len(x_test)):
    fd , hog_im = hog(x_test[i] , orientations=9 , pixels_per_cell = (8,8),
                     cells_per_block = (2,2) , visualize = True ,  multichannel = True)
    x_test_hog.append(fd)

x_test_hog = np.array(x_test_hog) 
prdct = my_model.predict(x_test_hog)
print(accuracy_score(y_test , prdct))

def convert(image):
    img=Image.open(image)
    img=img.resize((32,32),Image.ANTIALIAS)
    return img
matrix=convert("enter file path")

fd , hog_im = hog(matrix , orientations=9 , pixels_per_cell = (8,8), cells_per_block = (2,2) , visualize = True ,  multichannel = True)

prdct = my_model.predict(fd.reshape(1, -1))[0]
print(classes[prdct])