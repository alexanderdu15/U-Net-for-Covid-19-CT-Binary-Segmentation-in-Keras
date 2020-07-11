import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
import os, sys

def diceCoef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def diceCoefLoss(y_true, y_pred):
    return (1-diceCoef(y_true, y_pred))

def jaccardDistance(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1-jac) * smooth

def f1Score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

pathImgs, pathMasks = '', '' #enter the filepath of CT images and masks
imgRaw, maskRaw = nib.load(pathImgs), nib.load(pathMasks) #loading dataset from original file type

img, mask = np.asanyarray(imgRaw.dataobj), np.asanyarray(maskRaw.dataobj) #converting dataset to numpy ndarrays
print('Dataset loaded')

img, mask = img.reshape(512,512,100,-1), mask.reshape(512,512,100,-1) #adding channel dimension
img, mask = np.transpose(img,(2,0,1,3)), np.transpose(mask,(2,0,1,3)) #reordering arrays
mask[mask > 0] = 1 #binarizing masks
trainImgs, testImgs, trainMasks, testMasks = train_test_split(img, mask, test_size = 0.4) #splitting the dataset
print('Dataset split completed')
print('Tensor sizes:')
print('Training images:'+str(trainImgs.shape)+'\nTest images:'+str(testImgs.shape)+'\nTest masks:'+str(trainMasks.shape)+'\nTest masks:'+str(testMasks.shape))

dataGenArgs = dict(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1) #ImageDataGenerator arguments
imageGen, maskGen = keras.preprocessing.image.ImageDataGenerator(**dataGenArgs), keras.preprocessing.image.ImageDataGenerator(**dataGenArgs)

seed = 1
imageGen.fit(trainImgs, augment=True, seed=seed)
maskGen.fit(trainMasks, augment=True, seed=seed)

imageGenerator, maskGenerator = imageGen.flow(trainImgs, shuffle=False, batch_size = 8, seed=seed), maskGen.flow(trainMasks, shuffle=False, batch_size = 8, seed=seed)

trainGenerator = zip(imageGenerator, maskGenerator)

imgWidth = 512
imgHeight = 512
imgChannels = 1

#defining U-Net

inputs = keras.layers.Input((imgWidth, imgHeight, imgChannels), name = 'input')

c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c1-1")(inputs)
d1 = keras.layers.AlphaDropout(0.05, name = "d1")(c1)
c1 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c1-2")(d1)
p1 = keras.layers.MaxPooling2D((2,2), name = "p1")(c1)

c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c2-1")(p1)
d2 = keras.layers.AlphaDropout(0.05, name = "d2")(c2)
c2 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c2-2")(d2)
p2 = keras.layers.MaxPooling2D((2,2), name = "p2")(c2)

c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c3-1")(p2)
d3 = keras.layers.AlphaDropout(0, name = 'd3')(c3)
c3 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c3-2")(d3)
p3 = keras.layers.MaxPooling2D((2,2), name = "p3")(c3)

c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c4-1")(p3)
d4 = keras.layers.AlphaDropout(0.05, name = "d4")(c4)
c4 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c4-2")(d4)
p4 = keras.layers.MaxPooling2D((2,2), name = "p4")(c4)

c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c5-1")(p4)
d5 = keras.layers.AlphaDropout(0, name = "d5")(c5)
c5 = keras.layers.Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c5-2")(d5)
u6 = keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same', name = "t1")(c5)

u6 = keras.layers.concatenate([u6, c4], name = "cc1")
c6 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c6-1")(u6)
d6 = keras.layers.AlphaDropout(0, name = "d6")(c6)
c6 = keras.layers.Conv2D(128, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c6-2")(d6)
u7 = keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same', name = "t2")(c6)

u7 = keras.layers.concatenate([u7, c3], name = "cc2")
c7 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c7-1")(u7)
d7 = keras.layers.AlphaDropout(0.05, name = "d7")(c7)
c7 = keras.layers.Conv2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c7-2")(d7)
u8 = keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', name = "t3")(c7)

u8 = keras.layers.concatenate([u8, c2], name = "cc3")
c8 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c8-1")(u8)
d8 = keras.layers.AlphaDropout(0, name = "d8")(c8)
c8 = keras.layers.Conv2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c8-2")(d8)
u9 = keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same', name = "t4")(c8)

u9 = keras.layers.concatenate([u9, c1], name = "cc4")
c9 = keras.layers.Conv2D(16, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name = "c9-1")(u9)
d9 = keras.layers.AlphaDropout(0.05, name = "d9")(c9)
c10 = keras.layers.Conv2D(16, (1,1), strides=(1,1), padding='same', name = "c10")(d9)

outputs = keras.layers.Conv2D(1,(1,1), activation='sigmoid', name = "output")(c10)

model = keras.Model(inputs=[inputs], outputs=[outputs])

metrics = ['Accuracy', 'Precision', 'Recall', 'AUC', diceCoef, jaccardDistance, f1Score]

model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=0.0001), loss=diceCoefLoss , metrics=metrics)

model.summary()

model.load_weights('') #enter filepath for saved model weights

resutls = model.fit(trainGenerator, steps_per_epoch = 10, epochs = 200, verbose = 1) #train model

model.save_weights('') #enter filepath for saving model weights

model.evaluate(testImgs, testMasks, batch_size = 8) #evaluate model
