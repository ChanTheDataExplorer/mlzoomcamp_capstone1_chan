import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import os, os.path

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications.xception import decode_predictions

from tensorflow.keras.preprocessing.image import load_img

from collections import defaultdict
from hashlib import md5
from pathlib import Path

import PIL
from PIL import Image

import matplotlib.image as mpimg

import dataprep as dp


# Setting the dataset directories
dirs = {}

dirs['dataset_dir'] = './dataset'
dirs['raw_img_dir'] = './dataset/images'

print("Preparing the data")

#Check train.csv and test.csv
train_info = pd.read_csv(dirs['dataset_dir'] + '/train.csv', dtype = 'string')
test_info = pd.read_csv(dirs['dataset_dir'] + '/test.csv', dtype = 'string')

test_info.head()

# Details:
# * train.csv - contains the images to be used for training the model. It includes the label it belongs to
# * test.csv - contains the images to be used for testing the images which will be used for submission in the competition

# Then, we will use the dataprep.py file to:
# * remove duplicate images
# * sort the images into train and test
# * further separate the train folder to train and validation sets
#     * random shuffling with seed = 42

dp.call_class('./dataset')

print("Successful data preparation.")
print()

# Add the sorted train, validation, and test data
dirs['sorted_img_train'] = './dataset/sorted_images_train'
dirs['sorted_img_val'] = './dataset/sorted_images_val'
dirs['img_dir_test'] = './dataset/sorted_test'


# ### Training the Model
# * Train a 299x299 model
def make_model(input_size=150, learning_rate=0.01, size_inner=100,
               droprate=0.5):

    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(input_size, input_size, 3)
    )

    base_model.trainable = False

    #########################################

    inputs = keras.Input(shape=(input_size, input_size, 3))
    base = base_model(inputs, training=False)
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    inner = keras.layers.Dense(size_inner, activation='relu')(vectors)
    drop = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(6)(drop)
    
    model = keras.Model(inputs, outputs)
    
    #########################################

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

input_size = 299

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=10.0,
    height_shift_range=10.0,
    shear_range=10.0,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True,
)

train_ds = train_gen.flow_from_directory(
    dirs['sorted_img_train'],
    target_size=(input_size, input_size),
    batch_size=32
)


val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_ds = train_gen.flow_from_directory(
    dirs['sorted_img_val'],
    target_size=(input_size, input_size),
    batch_size=32,
    shuffle=False
)


checkpoint = keras.callbacks.ModelCheckpoint(
    'xception_v2_{epoch:02d}_{val_accuracy:.3f}.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max'
)

learning_rate = 0.001
size = 10
droprate = 0.2

model = make_model(
    input_size=input_size,
    learning_rate=learning_rate,
    size_inner=size,
    droprate=droprate
)

history = model.fit(train_ds, epochs=50, validation_data=val_ds,
                   callbacks=[checkpoint])

print("Finished training the model")
print("You can just retain the best model saved in the directory")