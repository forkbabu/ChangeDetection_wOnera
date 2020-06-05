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
from keras import backend as K
import pandas as pd
import cd_models as cdm
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, auc, roc_auc_score, roc_curve
import itertools


batch_size = 32
img_size = 128
channels = 13
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
    # raster1 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_1_rect/')
    raster1 = rnc.build_raster(dataset_dir + f + '/imgs_1_rect/')
    # raster2 = rnc.build_rasterRGB(dataset_dir + f + '/imgs_2_rect/')
    raster2 = rnc.build_raster(dataset_dir + f + '/imgs_2_rect/')
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
model = load_model(save_dir + model_name + '_deconv.h5', custom_objects={'weighted_bce_dice_loss': cdm.weighted_bce_dice_loss})
# model = load_model(save_dir + model_name + '_deconv.h5')
model.summary()

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
