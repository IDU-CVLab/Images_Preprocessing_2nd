#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 21:21:42 2023

@author: Kenan Morani using idu Workstation

"""
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


##################################################

##### Stage 1 : K-means clusreting based segmetnation

################################################
def extract_lungs(mask):
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def kmeans_segmentation(image):
    Z = image.reshape((-1,1))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image

def segment_and_extract_lungs(image):
    segmented_image = kmeans_segmentation(image)
    binary_mask = np.zeros(image.shape, dtype=np.uint8)
    binary_mask[segmented_image == segmented_image.min()] = 1
    binary_mask = extract_lungs(binary_mask)
    lung_extracted_image = np.zeros(image.shape, dtype=np.uint8)
    lung_extracted_image[binary_mask == 1] = image[binary_mask == 1]
    return lung_extracted_image

##### Modify here to run the code on all the required slices
input_folder = "/home/idu/Desktop/COV19D/train/non-covid"
output_folder = "/home/idu/Desktop/COV19D/train-seg1/non-covid"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        image_path = os.path.join(subdir, file)
        if '.jpg' in image_path:
            image = cv2.imread(image_path, 0)
            lung_extracted_image = segment_and_extract_lungs(image)
            subfolder_name = subdir.split('/')[-1]
            subfolder_path = os.path.join(output_folder, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            output_path = os.path.join(subfolder_path, file)
            cv2.imwrite(output_path, lung_extracted_image)

###########################################
##################Cropping the resulting images in the center to size 256x256 [optional]



import cv2
import numpy as np
import os

def crop_center(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy, startx:startx+cropx]

## Modify here to run the code on all the required slices
input_folder = "/home/idu/Desktop/COV19D/validation/non-covid"
output_folder = "/home/idu/Desktop/COV19D/val-seg1/non-covid"


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for subdir, dirs, files in os.walk(input_folder):
    for file in files:
        image_path = os.path.join(subdir, file)
        if '.jpg' in image_path:
            image = cv2.imread(image_path, 0)
            cropped_image = crop_center(image, 256, 256)
            subfolder_name = subdir.split('/')[-1]
            subfolder_path = os.path.join(output_folder, subfolder_name)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
            output_path = os.path.join(subfolder_path, file)
            cv2.imwrite(output_path, cropped_image)
            
            
######################################################

############ Stage2 : Classification Using a CNN model

######################################################

import numpy as np
import tensorflow as tf
#import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

def lr_schedule(epoch):
    lr = 0.1 * np.exp(-epoch)
    print('Learning rate: ', lr)
    return lr

h=w=224

def make_model():
   
    model = tf.keras.models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.3))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()


# compile the model

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 
                                  'accuracy'])

# create the data generator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

# load the training and validation data
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-seg1', ## Slices segmented using kmeans clustering
        target_size=(h, w),
        batch_size=128,
        color_mode='grayscale',
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/val-seg1', ## Slices segmented using kmeans clustering
        target_size=(h, w),
        batch_size=128,
        color_mode='grayscale',
        class_mode='binary')

# define early stopping, checkpoint and learning rate scheduler callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=17)
checkpoint = ModelCheckpoint('/home/idu/Desktop/COV19D/ChatGPT-saved-models/UNet-seg-sliceremove-cnn-class.h5', save_best_only=True, save_weights_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# train the model
history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50,
        callbacks=[early_stopping, lr_scheduler, checkpoint])


###### Evaluate models

model = make_model()

model.load_weights('/home/idu/Desktop/COV19D/ChatGPT-saved-models/kmeans-cluster-seg-cnn-classif.h5')

model.evaluate(validation_generator, batch_size=128)
