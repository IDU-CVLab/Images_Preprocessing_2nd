#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 12:54:22 2022

@author: idu
"""

## UNet Model for segmentation

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf
 
from tensorflow import keras
from skimage import morphology
from skimage.feature import canny
#from scipy import ndimage as ndi
from skimage import io
from skimage.exposure import histogram

from PIL import Image as im
import skimage
from skimage.filters import sobel

# Mounting my google drive#
#from google.colab import drive

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
from PIL import Image
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Stage 1 : UNET MODEL BUILDING #########
##^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""# Slicing and Saving"""

## The path here include the image volumes (308MB) and the lung masks (1MB) were extracted from the COVID-19 CT segmentation dataset
dataInputPath = '/home/idu/Desktop/COV19D/segmentation/volumes'
imagePathInput = os.path.join(dataInputPath, 'img/') ## Image volumes were exctracted to this subfolder
maskPathInput = os.path.join(dataInputPath, 'mask/') ## lung masks were exctracted to this subfolder

# Preparing the outputpath for slicing the CT volume from the above data
dataOutputPath = '/home/idu/Desktop/COV19D/segmentation/slices/'
imageSliceOutput = os.path.join(dataOutputPath, 'img/') ## Image volume slices will be placed here
maskSliceOutput = os.path.join(dataOutputPath, 'mask/') ## Annotated masks slices will be placed here

# Slicing only in Z direction
# Slices in Z direction shows the required lung area
SLICE_X = False
SLICE_Y = False
SLICE_Z = True

SLICE_DECIMATE_IDENTIFIER = 3

# Choosing normalization boundaries suitable from the chosen images
HOUNSFIELD_MIN = -1020
HOUNSFIELD_MAX = 2995
HOUNSFIELD_RANGE = HOUNSFIELD_MAX - HOUNSFIELD_MIN

# Normalizing the images
def normalizeImageIntensityRange (img):
    img[img < HOUNSFIELD_MIN] = HOUNSFIELD_MIN
    img[img > HOUNSFIELD_MAX] = HOUNSFIELD_MAX
    return (img - HOUNSFIELD_MIN) / HOUNSFIELD_RANGE

#nImg = normalizeImageIntensityRange(img)
#np.min(nImg), np.max(nImg), nImg.shape, type(nImg)

# Reading image or mask volume
def readImageVolume(imgPath, normalize=True):
    img = nib.load(imgPath).get_fdata()
    if normalize:
        return normalizeImageIntensityRange(img)
    else:
        return img
    
#readImageVolume(imgPath, normalize=False)
#readImageVolume(maskPath, normalize=False)

# Slicing image in all directions and save
def sliceAndSaveVolumeImage(vol, fname, path):
    (dimx, dimy, dimz) = vol.shape
    print(dimx, dimy, dimz)
    cnt = 0
    if SLICE_X:
        cnt += dimx
        print('Slicing X: ')
        for i in range(dimx):
            saveSlice(vol[i,:,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_x', path)
            
    if SLICE_Y:
        cnt += dimy
        print('Slicing Y: ')
        for i in range(dimy):
            saveSlice(vol[:,i,:], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_y', path)
            
    if SLICE_Z:
        cnt += dimz
        print('Slicing Z: ')
        for i in range(dimz):
            saveSlice(vol[:,:,i], fname+f'-slice{str(i).zfill(SLICE_DECIMATE_IDENTIFIER)}_z', path)
    return cnt

# Saving volume slices to file
def saveSlice (img, fname, path):
    img = np.uint8(img * 255)
    fout = os.path.join(path, f'{fname}.png')
    cv2.imwrite(fout, img)
    print(f'[+] Slice saved: {fout}', end='\r')

# Reading and processing image volumes for TEST images
for index, filename in enumerate(sorted(glob.iglob(imagePathInput+'*.nii.gz'))):
    img = readImageVolume(filename, True)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 't'+str(index), imageSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# Reading and processing image mask volumes for TEST masks
for index, filename in enumerate(sorted(glob.iglob(maskPathInput+'*.nii.gz'))):
    img = readImageVolume(filename, False)
    print(filename, img.shape, np.sum(img.shape), np.min(img), np.max(img))
    numOfSlices = sliceAndSaveVolumeImage(img, 't'+str(index), maskSliceOutput)
    print(f'\n{filename}, {numOfSlices} slices created \n')

# Exploring the data 
imgPath = os.path.join(imagePathInput, '1.nii.gz')
img = nib.load(imgPath).get_fdata()
np.min(img), np.max(img), img.shape, type(img)


maskPath = os.path.join(maskPathInput, '1.nii.gz')
mask = nib.load(maskPath).get_fdata()
np.min(mask), np.max(mask), mask.shape, type(mask)

# Showing Mask slice
imgSlice = mask[:,:,20]
plt.imshow(imgSlice, cmap='gray')
plt.show()

# Showing Corresponding Image slice
imgSlice = img[:,:,20]
plt.imshow(imgSlice, cmap='gray')
plt.show()

"""# Training and testing Generator"""

# Define constants
SEED = 42

### Setting the training and testing dataset for validation of the proposed VGG16-UNET model
### All t0&t1 Z-sliced slices (images and masks) were used for testing

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
SIZE = IMAGE_HEIGHT = IMAGE_HEIGHT
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

#### Splitting the data into training and test sets happen manually
### t0 volumes and masks were chosen as test sets
data_dir = '/home/idu/Desktop/COV19D/segmentation/slices/'
data_dir_train = os.path.join(data_dir, 'training')
# The images should be stored under: "data/slices/training/img/img"
data_dir_train_image = os.path.join(data_dir_train, 'img')
# The images should be stored under: "data/slices/training/mask/img"
data_dir_train_mask = os.path.join(data_dir_train, 'mask')

data_dir_test = os.path.join(data_dir, 'test')
# The images should be stored under: "data/slices/test/img/img"
data_dir_test_image = os.path.join(data_dir_test, 'img')
# The images should be stored under: "data/slices/test/mask/img"
data_dir_test_mask = os.path.join(data_dir_test, 'mask')


def create_segmentation_generator_train(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255,
                      featurewise_center=True,
                      featurewise_std_normalization=True,
                      rotation_range=90,
                      width_shift_range=0.2,
                      height_shift_range=0.2,
                      zoom_range=0.3
                        )
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

# Remember not to perform any image augmentation in the test generator!
def create_segmentation_generator_test(img_path, msk_path, BATCH_SIZE):
    data_gen_args = dict(rescale=1./255)
    datagen = ImageDataGenerator(**data_gen_args)
    
    img_generator = datagen.flow_from_directory(img_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    msk_generator = datagen.flow_from_directory(msk_path, target_size=IMG_SIZE, class_mode=None, color_mode='grayscale', batch_size=BATCH_SIZE, seed=SEED)
    return zip(img_generator, msk_generator)

train_generator = create_segmentation_generator_train(data_dir_train_image, data_dir_train_mask, BATCH_SIZE_TRAIN)
test_generator = create_segmentation_generator_test(data_dir_test_image, data_dir_test_mask, BATCH_SIZE_TEST)


NUM_TRAIN = 745
NUM_TEST = 84

NUM_OF_EPOCHS = 20

def display(display_list):
    plt.figure(figsize=(15,15))
    
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        display([image[0], mask[0]])

show_dataset(train_generator, 4)

EPOCH_STEP_TRAIN = 6*NUM_TRAIN // BATCH_SIZE_TRAIN
EPOCH_STEP_TEST = NUM_TEST // BATCH_SIZE_TEST

"""# UNet Model"""

from tensorflow.keras.layers import Conv2D, Dropout, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

SIZE = 224
IMAGE_HEIGHT = SIZE
IMAGE_WIDTH = SIZE

#### UNET NETWORK

# Building 2D-UNET model
def unet(n_levels, initial_features=32, n_blocks=2, kernel_size=3, pooling_size=2, in_channels=1, out_channels=1):
    inputs = keras.layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, in_channels))
    x = inputs
    
    convpars = dict(kernel_size=kernel_size, activation='relu', padding='same')
    
    #downstream
    skips = {}
    for level in range(n_levels):
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
        if level < n_levels - 1:
            skips[level] = x
            x = keras.layers.MaxPool2D(pooling_size)(x)
            
    # upstream
    for level in reversed(range(n_levels-1)):
        x = keras.layers.Conv2DTranspose(initial_features * 2 ** level, strides=pooling_size, **convpars)(x)
        x = keras.layers.Concatenate()([x, skips[level]])
        for _ in range(n_blocks):
            x = keras.layers.Conv2D(initial_features * 2 ** level, **convpars)(x)
            
    # output
    activation = 'sigmoid' if out_channels == 1 else 'softmax'
    x = keras.layers.Conv2D(out_channels, kernel_size=1, activation=activation, padding='same')(x)
    
    return keras.Model(inputs=[inputs], outputs=[x], name=f'UNET-L{n_levels}-F{initial_features}')

UNet_model = unet(3)

UNet_model.summary()


UNet_model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[#tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                      tf.keras.metrics.MeanIoU(num_classes = 2),
                      'accuracy'])

UNet_model.fit_generator(generator=train_generator, 
                    steps_per_epoch=EPOCH_STEP_TRAIN, 
                    validation_data=test_generator, 
                    validation_steps=EPOCH_STEP_TEST,
                    epochs=NUM_OF_EPOCHS)

#Saving the model
UNet_model = keras.models.load_model('/home/idu/Desktop/COV19D/segmentation/UNet_model.h5')

# Displaying images
def display(display_list):
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='gray')
    plt.show()

def show_dataset(datagen, num=1):
    for i in range(0, num):
        image,mask = next(datagen)
        display((image[0], mask[0]))


from tensorflow.keras.metrics import MeanIoU 

path = '/home/idu/Desktop/COV19D/segmentation/Segmentation Results/'
def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = UNet_model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])        
        num_classes = 2
        IOU_keras = MeanIoU(num_classes=num_classes)  
        IOU_keras.update_state(mask[0], pred_mask)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
        print(values) 
        

show_prediction(test_generator, 12)


results = UNet_model.evaluate(test_generator, batch_size=32)
print("test loss, test acc:", results)
        
  ## Segmenting images based on k-means clustering and exctracting lung regions and saving them 
#in the same directory as the original images (in .png format)
  
path = '/home/idu/Desktop/COV19D/segmentation/Segmentation Results/'
def show_prediction(datagen, num=1):
    for i in range(0,num):
        image,mask = next(datagen)
        pred_mask = UNet_model.predict(image)[0] > 0.5
        display([image[0], mask[0], pred_mask])        
        num_classes = 2
        IOU_keras = MeanIoU(num_classes=num_classes)  
        IOU_keras.update_state(mask[0], pred_mask)
        print("Mean IoU =", IOU_keras.result().numpy())

        values = np.array(IOU_keras.get_weights()).reshape(num_classes, num_classes)
        print(values) 
        

show_prediction(test_generator, 12)

### Finish First Stage ---- KENAN MORANI