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
channels = 3
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
     raster1 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_1_rect/')
    #raster1 = rnc.build_raster(dataset_dir + f + '/imgs_1_rect/')
     raster2 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_2_rect/')
    #raster2 = rnc.build_raster(dataset_dir + f + '/imgs_2_rect/')
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
#history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, class_weight = weights, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)
history = model.fit(inputs, labels, batch_size=batch_size, epochs=epochs, class_weight = None, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)


# history = model.fit(inputs, 5*[labels], batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)], shuffle=True, verbose=1)

# Save the history for accuracy/loss plotting
history_save = pd.DataFrame(history.history).to_hdf(save_dir + history_name + "_deconv.h5", "history", append=False)


# coding: utf-8

"""
Evaluate scores of a CD UNet/UNet++ trained model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Wed Jan 08, 2020
"""


import os
import read_and_crop as rnc
import numpy as np
from libtiff import TIFF
import tensorflow as tf
import pandas as pd
import cd_models as cdm
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, auc, roc_auc_score, roc_curve
import itertools


batch_size = 32
img_size = 128
channels = 3
stride = 128
classes = 1
dataset_dir = '../OneraDataset_Images/'
labels_dir = '../OneraDataset_TrainLabels/'
save_dir = '../models/'
plot_dir = '../plots/'
cm_dir = plot_dir + 'confusion_matrix/'
roc_dir = plot_dir + 'roc/'
score_dir = '../scores/'
model_name = 'EF-UNet_128-128_sklw-bce'
class_names = ['unchange', 'change']


# Get the list of folders to open to get rasters
folders = rnc.get_folderList(dataset_dir + 'train.txt')

# Build rasters, pad them and crop them to get the input images
train_images = []
num_crops = []
padded_shapes = []

for f in folders:
     raster1 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_1_rect/')
    #raster1 = rnc.build_raster(dataset_dir + f + '/imgs_1_rect/')
     raster2 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_2_rect/')
    #raster2 = rnc.build_raster(dataset_dir + f + '/imgs_2_rect/')
    raster = np.concatenate((raster1,raster2), axis=2)
    padded_raster = rnc.pad(raster, img_size)
    shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
    padded_shapes.append(shape)
    crops = rnc.crop(padded_raster, img_size, stride)
    num_crops.append(len(crops))
    train_images = train_images + crops

# Read change maps to get the ground truths
train_labels = []
unpadded_shapes = []
for f in folders:
    cm = TIFF.open(labels_dir + f + '/cm/' + f + '-cm.tif').read_image()
    cm = np.expand_dims(cm, axis=2)
    cm -= 1 # the change map has values 1 for unchange and 2 for change ---> scale back to 0 and 1
    unpadded_shapes.append(cm.shape)
    cm = cm.flatten()
    train_labels.append(cm)

# Create inputs and labels for the Neural Network
inputs = np.asarray(train_images)
y_true = np.asarray(train_labels)
print(y_true.shape)

# Load the model

# model = load_model(save_dir + model_name + '_deconv.h5')
print("SCORE")

# Perform inference
results = model.predict(inputs)

# Build unpadded change maps 
index = 0
y_pred = []
y_pred_r = [] # rounded predictions

for i in range(len(folders)):
    crops = num_crops[i]
    padded_cm = rnc.uncrop(padded_shapes[i], results[index:index+crops], img_size, stride)
    cm = rnc.unpad(unpadded_shapes[i], padded_cm)
    cm_r = np.rint(cm)
    cm_r = cm_r.flatten()
    cm = cm.flatten()
    y_pred.append(cm)
    y_pred_r.append(cm_r)
    index += crops

# Flatten results
y_pred = [item for sublist in y_pred for item in sublist]
y_pred_r = [item for sublist in y_pred_r for item in sublist]
y_true = [item for sublist in train_labels for item in sublist]

# Print scores
f = open(score_dir + model_name + '_scores.txt',"w+")

f.write("Precision: %f\n" % precision_score(y_true, y_pred_r))
f.write("Recall: %f\n" % recall_score(y_true, y_pred_r))
f.write("F1: %f\n" % f1_score(y_true, y_pred_r))
f.write("Balanced Accuracy: %f\n" % balanced_accuracy_score(y_true, y_pred_r))
f.write("Accuracy: %f\n" % accuracy_score(y_true, y_pred_r))
f.write("ROC AUC: %f\n" % roc_auc_score(y_true, y_pred)) # ROC AUC Score needs non-rounded predictions!

f.close()

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=13, y=1.04)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.ylabel('True class', labelpad=10, fontsize=13)
    plt.xlabel('Predicted class', labelpad=10, fontsize=13)
    plt.tight_layout()


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_true, y_pred_r, normalize='true')
np.set_printoptions(precision=2)

if not os.path.exists(cm_dir):
    os.mkdir(cm_dir)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.savefig(cm_dir + model_name + '.pdf', format='pdf')
plt.show()


# ROC curve
fpr = []
tpr = []

fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

if not os.path.exists(roc_dir):
    os.mkdir(roc_dir)


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='orangered', lw=1.5, label=model_name+' (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='royalblue', lw=1.5, linestyle='--')
plt.plot([0, 1], [1, 1], color='black', lw=0.7, linestyle=':')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(roc_dir + model_name + '_roc.pdf', format='pdf')
plt.show()

# Finally, save trp and fpr to file for roc curve comparison  

rates = ['tpr', 'fpr']
datas = np.stack((tpr,fpr))
df = pd.DataFrame(datas, index = rates)
df.to_hdf(roc_dir + model_name + '_rates_deconv.h5',"rates",complevel=0)


# coding: utf-8

"""
Perform inference with a CD UNet/UNet++ model for Onera Dataset, available @ http://dase.grss-ieee.org

@Author: Tony Di Pilato

Created on Fri Dec 13, 2019
"""

import os
import read_and_crop as rnc
import numpy as np
from libtiff import TIFF
import tensorflow as tf
import cd_models as cdm
from keras.models import load_model
import matplotlib.pyplot as plt
import random


img_size = 128
channels = 3
stride = 128
classes = 1

dataset_dir = '../OneraDataset_Images/'
labels_dir = '../OneraDataset_TrainLabels/'

save_dir = '../models/'
#model_name = 'EF_UNet_bce-256_ol64'
model_name = 'EF-UNet_128-128_sklw-bce'

infres_dir = '../results/'
history_name = model_name + '_history'

# Get the list of folders to open to get rasters
# folders = rnc.get_folderList(dataset_dir + 'test.txt')
folders = rnc.get_folderList(dataset_dir + 'train.txt')

# Select a folder, build raster, pad it and crop it to get the input images
# f = random.choice(folders)
f = 'rennes'

raster1 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_1_rect/')
raster2 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_2_rect/')
raster = np.concatenate((raster1,raster2), axis=2)
padded_raster = rnc.pad(raster, img_size)
test_image = rnc.crop(padded_raster, img_size, stride)

# Create inputs for the Neural Network
inputs = np.asarray(test_image)
print("INFERENCE")

# Perform inference
results = model.predict(inputs)

# Build the complete change map
# results = results[4] # This should be used if DS enabled
shape = (padded_raster.shape[0], padded_raster.shape[1], classes)
padded_cm = rnc.uncrop(shape, results, img_size, stride)
cm = rnc.unpad(raster.shape, padded_cm)

cm = np.squeeze(cm)
cm = np.rint(cm) # we are only interested in change/unchange

res_dir = infres_dir + f

if not os.path.exists(res_dir):
    os.mkdir(res_dir)

# Plot and save the change map
fig = plt.imshow(cm, cmap='gray')
plt.axis('off')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig(res_dir + '/' + model_name + '.png', bbox_inches = 'tight', pad_inches = 0)

