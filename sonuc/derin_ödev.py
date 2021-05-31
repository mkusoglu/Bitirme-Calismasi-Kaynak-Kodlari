# -*- coding: utf-8 -*-
"""
Created on Tue May 18 20:47:11 2021

@author: MustafaKuşoğlu
"""
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import np_utils
from keras import losses
from keras.datasets import cifar100
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
(x_train,y_train),(x_test,y_test)=cifar100.load_data()


index = np.where(y_train == 23)
X_train = x_train[index[0]]
Y_train = y_train[index[0]]

index1 = np.where(y_train == 24)
X1_train = x_train[index1[0]]
Y1_train = y_train[index1[0]]

index2 = np.where(y_train == 37)
X2_train = x_train[index2[0]]
Y2_train = y_train[index2[0]]

index3 = np.where(y_train == 40)
X3_train = x_train[index3[0]]
Y3_train = y_train[index3[0]]

index4 = np.where(y_train == 80)
X4_train = x_train[index4[0]]
Y4_train = y_train[index4[0]]
#
index5 = np.where(y_test == 23)
X_test = x_test[index5[0]]
Y_test = y_test[index5[0]]

index6 = np.where(y_test == 24)
X1_test = x_test[index6[0]]
Y1_test = y_test[index6[0]]

index7 = np.where(y_test == 37)
X2_test = x_test[index7[0]]
Y2_test = y_test[index7[0]]

index8 = np.where(y_test == 40)
X3_test = x_test[index8[0]]
Y3_test = y_test[index8[0]]

index9= np.where(y_test == 80)
X4_test = x_test[index9[0]]
Y4_test = y_test[index9[0]]

r=np.concatenate((X_train, X1_train), axis=0)
r=np.concatenate((r, X2_train), axis=0)
r=np.concatenate((r, X3_train), axis=0)
r=np.concatenate((r, X4_train), axis=0)

r2=np.concatenate((Y_train, Y_train), axis=0)
r2=np.concatenate((r2, Y2_train), axis=0)
r2=np.concatenate((r2, Y3_train), axis=0)
r2=np.concatenate((r2, Y4_train), axis=0)

r3=np.concatenate((X_test, X1_test), axis=0)
r3=np.concatenate((r3, X2_test), axis=0)
r3=np.concatenate((r3, X3_test), axis=0)
r3=np.concatenate((r3, X4_test), axis=0)

r4=np.concatenate((Y_test, Y1_test), axis=0)
r4=np.concatenate((r4, Y2_test), axis=0)
r4=np.concatenate((r4, Y3_test), axis=0)
r4=np.concatenate((r4, Y4_test), axis=0)
"""
den1=np.concatenate((Y_train, Y1_train), axis=0)
den2=np.concatenate((Y2_train, Y3_train), axis=0)
"""


x_train=r
y_train=r2
x_test=r3
y_test=r4

#x_train, y_train = shuffle(x_train, y_train, random_state=0)
#x_test, y_test = shuffle(x_test, y_test, random_state=0)

x_train=x_train / 255.0
x_test=x_test/255.0

yedek=y_train

yedek=np.where(yedek==23, 1, yedek)
yedek=np.where(yedek==24, 2, yedek)
yedek=np.where(yedek==37, 3, yedek)
yedek=np.where(yedek==40, 4, yedek)
yedek=np.where(yedek==80, 5, yedek)

yedek_c = to_categorical(yedek)
yedek_c=np.delete(yedek_c,0,1)
y_train=yedek_c

y_test=np.where(y_test==23,1,y_test)
y_test=np.where(y_test==24,2,y_test)
y_test=np.where(y_test==37,3,y_test)
y_test=np.where(y_test==40,4,y_test)
y_test=np.where(y_test==80,5,y_test)
y_test = to_categorical(y_test)
y_test=np.delete(y_test,0,1)

x_train, y_train = shuffle(x_train, y_train, random_state=0)
x_test, y_test = shuffle(x_test, y_test, random_state=0)

model = models.Sequential()

model.add(layers.Conv2D(32,(5, 5),
                        #strides=(2,2),
                        padding='same',#'valid'
                        activation='relu',
                        input_shape=(32,32, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, 
                        (3, 3), 
                        padding='valid',
                        activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, 
                        (3, 3), 
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))


model.add(layers.Conv2D(64, 
                        (3, 3), 
                        padding='same',
                        activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(5, activation='softmax'))


model.summary()

from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy', #losses.sparse_categorical_crossentropy
              metrics=['accuracy'])

model.fit(x_train, 
          y_train, 
          epochs=100, 
          batch_size=32,#32, 4 ten daha iyi oldu
          validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("test_acc=",test_acc)