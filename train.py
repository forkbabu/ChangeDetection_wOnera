# coding: utf-8

"""
Train a CD UNet/UNet++ model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""


import os
import read_and_crop as rnc
import numpy as np
from libtiff import TIFF
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


import pandas as pd
import freeze
import cd_models as cdm
import argparse
from sklearn.utils import class_weight


parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=128)
parser.add_argument('--stride', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--loss', type=str, default='bce')


args = parser.parse_args()

batch_size = 32
img_size = args.size
channels = 13
stride = args.stride
classes = 1
epochs = args.epochs
dataset_dir = '../OneraDataset_Images/'
labels_dir = '../OneraDataset_TrainLabels/'
save_dir = '../models/'
frozen_dir = save_dir + 'frozen_models/'
loss = args.loss
model_name = 'EF-UNet_'+str(img_size)+'-'+str(stride)+'_'+'sklw-'+str(loss)
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
folders = rnc.get_folderList(dataset_dir + 'train.txt')

# Build rasters, pad them and crop them to get the input images
train_images = []
for f in folders:
    # raster1 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_1_rect/')
    raster1 = rnc.build_raster(dataset_dir + f + '/imgs_1_rect/')
    # raster2 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_2_rect/')
    raster2 = rnc.build_raster(dataset_dir + f + '/imgs_2_rect/')
    raster = np.concatenate((raster1,raster2), axis=2)
    padded_raster = rnc.pad(raster, img_size)
    train_images = train_images + rnc.crop(padded_raster, img_size, stride)    

# Read change maps, pad them and crop them to get the ground truths
train_labels = []
for f in folders:
    cm = TIFF.open(labels_dir + f + '/cm/' + f + '-cm.tif').read_image()
    cm = np.expand_dims(cm, axis=2)
    cm -= 1 # the change map has values 1 for no change and 2 for change ---> scale back to 0 and 1
    padded_cm = rnc.pad(cm, img_size)
    train_labels = train_labels + rnc.crop(padded_cm, img_size, stride)

# Create inputs and labels for the Neural Network
inputs = np.asarray(train_images)
labels = np.asarray(train_labels)


# Compute class weights
flat_labels = np.reshape(labels,[-1])
weights = 2*class_weight.compute_class_weight('balanced', np.unique(flat_labels), flat_labels)
print("**** Weights: ", weights)

# Create the model
model = cdm.EF_UNet([img_size,img_size,2*channels], classes, loss)
model.summary()

# Train the model
history = model.fit(inputs, labels.any(), batch_size=batch_size, epochs=epochs, class_weight=weights, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
# history = model.fit(inputs, 5*[labels], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save the history for accuracy/loss plotting
history_save = pd.DataFrame(history.history).to_hdf(save_dir + history_name + ".h5", "history", append=False)

# Save model and weights
model.save(save_dir + model_name + ".h5")
print('Trained model saved @ %s ' % save_dir)

# Save frozen graph
#frozen_graph = freeze.freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, frozen_dir, model_name + ".pbtxt", as_text=True)
#tf.train.write_graph(frozen_graph, frozen_dir, model_name + ".pb", as_text=False)

#print('Frozen model saved @ %s ' % frozen_dir)
