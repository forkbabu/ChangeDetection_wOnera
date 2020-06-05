# coding: utf-8

"""
@Author: Sayantan Das

Created on Thu, June 4, 2020
"""

import math
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Conv2DTranspose, concatenate,Layer
from tensorflow.keras.models import Model
from tf_deconv import FastDeconv2D


def dice_coef(y_true, y_pred, smooth=1, weight=0.5):
    y_true = y_true[:, :, :, -1]
    y_pred = y_pred[:, :, :, -1]
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + weight * K.sum(y_pred)
    # K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return ((2. * intersection + smooth) / (union + smooth))  # not working better using mean


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def weighted_bce_dice_loss(y_true,y_pred):
    class_loglosses = K.mean(K.binary_crossentropy(y_true, y_pred), axis=[0, 1, 2])

    class_weights = [0.1, 0.9]
    weighted_bce = K.sum(class_loglosses * K.constant(class_weights))
    # return K.weighted_binary_crossentropy(y_true, y_pred,pos_weight) + 0.35 * (self.dice_coef_loss(y_true, y_pred)) #not work
    return weighted_bce + 0.5 * (dice_coef_loss(y_true, y_pred))


def UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3):   
    ksize = kernel_size
    x = FastDeconv2D(in_channels=2,out_channels=nb_filter, kernel_size=(ksize,ksize), padding='same',activation='selu')(input_tensor)
    x = FastDeconv2D(in_channels=nb_filter,out_channels=nb_filter, kernel_size=(ksize,ksize), padding='same',activation='selu')(x)
    x = BatchNormalization(name='bn' + stage)(x)

    return x


def EF_UNet(input_shape, classes=1, loss='bce'):
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = Input(shape=input_shape, name='input') #batchsize,128,128,26#
    

    conv1 = UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4])
    
    # Right side of the U-Net
    up1 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = concatenate([conv4,up1], axis=bn_axis)
    conv6 = UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3])

    up2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = concatenate([conv3,up2], axis=bn_axis)
    conv7 = UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2])

    up3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = concatenate([conv2,up3], axis=bn_axis)
    conv8 = UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1])
    
    up4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = concatenate([conv1,up4], axis=bn_axis)
    conv9 = UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0])

    # Output layer of the U-Net with a softmax activation
    output = Conv2D(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv9)

    model = Model(inputs, output)

    if loss == 'bce':
        loss = 'binary_crossentropy'
    elif loss == 'wbce':
        loss = weighted_bce_dice_loss

    model.compile(optimizer=Adam(lr=1e-4), loss = loss, metrics = ['accuracy'])    
    
    return model
