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

pathImgs, pathMasks = 'tr_im.nii.gz', 'tr_mask.nii.gz' #enter the filepath of CT images and masks
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

dataGenArgs = dict(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1) #ImageDataGenerator arguments
imageGen, maskGen = keras.preprocessing.image.ImageDataGenerator(**dataGenArgs), keras.preprocessing.image.ImageDataGenerator(**dataGenArgs)

seed = 30
imageGen.fit(trainImgs, augment=True, seed=seed)
maskGen.fit(trainMasks, augment=True, seed=seed)

imageGenerator, maskGenerator = imageGen.flow(trainImgs, shuffle=False, batch_size = 8, seed=seed), maskGen.flow(trainMasks, shuffle=False, batch_size = 8, seed=seed)

trainGenerator = zip(imageGenerator, maskGenerator)
