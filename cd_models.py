# coding: utf-8

"""
@Author: Sayantan Das

Created on Thu, June 4, 2020
"""

import math

#from tensorflow.python.keras.layers.convolutional import Conv
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Conv2DTranspose, concatenate,Layer
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils import conv_utils

class BiasHeUniform(tf.keras.initializers.VarianceScaling):
    def __init__(self, seed=None):
        super(BiasHeUniform, self).__init__(scale=1. / 3., mode='fan_in', distribution='uniform', seed=seed)


# iteratively solve for inverse sqrt of a matrix
def isqrt_newton_schulz_autograd(A: tf.Tensor, numIters: int):
    dim = tf.shape(A)[0]
    normA = tf.norm(A, ord='fro', axis=[0, 1])
    Y = A / normA

    with tf.device(A.device):
        I = tf.eye(dim, dtype=A.dtype)
        Z = tf.eye(dim, dtype=A.dtype)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - tf.matmul(Z, Y))
        Y = tf.matmul(Y, T)
        Z = tf.matmul(T, Z)

    A_isqrt = Z / tf.sqrt(normA)
    return A_isqrt


def isqrt_newton_schulz_autograd_batch(A: tf.Tensor, numIters: int):
    Ashape = tf.shape(A)  # [batch, _, C]
    batchSize, dim = Ashape[0], Ashape[-1]

    normA = tf.reshape(A, (batchSize, -1))
    normA = tf.norm(normA, ord=2, axis=1)
    normA = tf.reshape(normA, [batchSize, 1, 1])

    Y = A / normA

    with tf.device(A.device):
        I = tf.expand_dims(tf.eye(dim, dtype=A.dtype), 0)
        Z = tf.expand_dims(tf.eye(dim, dtype=A.dtype), 0)

        I = tf.broadcast_to(I, Ashape)
        Z = tf.broadcast_to(Z, Ashape)

    for i in range(numIters):
        T = 0.5 * (3.0 * I - tf.matmul(Z, Y))
        Y = tf.matmul(Y, T)
        Z = tf.matmul(T, Z)

    A_isqrt = Z / tf.sqrt(normA)

    return A_isqrt



class FastDeconv2D(tf.keras.layers.Layer):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='valid', dilation_rate=1,
                 activation=None, use_bias=True, groups=1, eps=1e-5, n_iter=5, momentum=0.1, block=64,
                 sampling_stride=3, freeze=False, freeze_iter=100, kernel_initializer='he_uniform',
                 bias_initializer=BiasHeUniform(), **kwargs):
        self.in_channels = in_channels
        self.groups = groups
        self.momentum = momentum
        self.n_iter = n_iter
        self.eps = eps
        self.counter = 0
        self.track_running_stats = True

        if in_channels % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '.format(self.groups,
                                                                                       in_channels))
        if out_channels is not None and out_channels % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(groups, out_channels))

        super(FastDeconv2D, self).__init__(
            2, out_channels, kernel_size, stride, padding, dilation_rate=dilation_rate,
            activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer, **kwargs
        )

        if block > in_channels:
            block = in_channels
        else:
            if in_channels % block != 0:
                block = math.gcd(block, in_channels)
                print("`in_channels` not divisible by `block`, computing new `block` value as %d" % (block))

        if groups > 1:
            block = in_channels // groups

        self.block = block

        kernel_size_int_0 = kernel_size[0] if type(kernel_size) in (list, tuple) else kernel_size
        kernel_size_int_1 = kernel_size[1] if type(kernel_size) in (list, tuple) else kernel_size
        self.num_features = kernel_size_int_0 * kernel_size_int_1 * block

        if self.groups == 1:
            self.running_mean = tf.Variable(tf.zeros(self.num_features), trainable=False, dtype=self.dtype)
            self.running_deconv = tf.Variable(tf.eye(self.num_features), trainable=False, dtype=self.dtype)
        else:
            self.running_mean = tf.Variable(tf.zeros(kernel_size_int_0 * kernel_size_int_1 * in_channels),
                                            trainable=False, dtype=self.dtype)
            deconv_buff = tf.eye(self.num_features)
            deconv_buff = tf.expand_dims(deconv_buff, axis=0)
            deconv_buff = tf.tile(deconv_buff, [in_channels // block, 1, 1])
            self.running_deconv = tf.Variable(deconv_buff, trainable=False, dtype=self.dtype)

        stride_int = stride[0] if type(stride) in (list, tuple) else stride
        self.sampling_stride = sampling_stride * stride_int
        self.counter = tf.Variable(tf.convert_to_tensor(0, dtype=tf.int64), trainable=False)
        self.freeze_iter = freeze_iter
        self.freeze = freeze

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()

        # change rank based on 3d or 4d tensor input
        ndim = len(input_shape)

        self.input_spec = tf.keras.layers.InputSpec(min_ndim=3,
                                                    max_ndim=4,
                                                    axes={channel_axis: input_channel})

        self._build_conv_op_input_shape = input_shape
        self._build_input_channel = input_channel
        self._padding_op = self._get_padding_op()
        self._conv_op_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)
        self.built = True

    @tf.function(experimental_compile=False)
    def call(self, x, training=None):
        x_shape = tf.shape(x)
        N, H, W, C = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        block = self.block
        frozen = self.freeze and (self.counter > self.freeze_iter)

        if training and self.track_running_stats:
            counter = self.counter + 1
            counter = counter % (self.freeze_iter * 10)
            self.counter.assign(counter)

        if training and (not frozen):

            # 1. im2col: N x cols x pixels -> N*pixles x cols
            if self.kernel_size[0] > 1:
                # [N, L, L, C * K^2]
                X = tf.image.extract_patches(x,
                                             sizes=[1] + list(self.kernel_size) + [1],
                                             strides=[1, self.sampling_stride, self.sampling_stride, 1],
                                             rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
                                             padding=str(self.padding).upper())

                X = tf.reshape(X, [N, -1, C * self.kernel_size[0] * self.kernel_size[1]])  # [N, L^2, C * K^2]

            else:
                # channel wise ([N, H, W, C] -> [N * H * W, C] -> [N * H / S * W / S, C]
                X = tf.reshape(x, [-1, C])[::self.sampling_stride ** 2, :]

            if self.groups == 1:
                # (C//B*N*pixels,k*k*B)
                X = tf.reshape(X, [-1, self.num_features, C // block])
                X = tf.transpose(X, [0, 2, 1])
                X = tf.reshape(X, [-1, self.num_features])
            else:
                X_shape_ = tf.shape(X)
                X = tf.reshape(X, [-1, X_shape_[-1]])  # [N, L^2, C * K^2] -> [N * L^2, C * K^2]

            # 2. subtract mean
            X_mean = tf.reduce_mean(X, axis=0)
            X = X - tf.expand_dims(X_mean, axis=0)

            # 3. calculate COV, COV^(-0.5), then deconv
            if self.groups == 1:
                scale = tf.cast(tf.shape(X)[0], X.dtype)
                Id = tf.eye(X.shape[1], dtype=X.dtype)
                # addmm op
                Cov = self.eps * Id + (1. / scale) * tf.matmul(tf.transpose(X), X)
                deconv = isqrt_newton_schulz_autograd(Cov, self.n_iter)
            else:
                X = tf.reshape(X, [-1, self.groups, self.num_features])
                X = tf.transpose(X, [1, 0, 2])  # [groups, -1, num_features]

                Id = tf.eye(self.num_features, dtype=X.dtype)
                Id = tf.broadcast_to(Id, [self.groups, self.num_features, self.num_features])

                scale = tf.cast(tf.shape(X)[1], X.dtype)
                Cov = self.eps * Id + (1. / scale) * tf.matmul(tf.transpose(X, [0, 2, 1]), X)

                deconv = isqrt_newton_schulz_autograd_batch(Cov, self.n_iter)

            if self.track_running_stats:
                running_mean = self.momentum * X_mean + (1. - self.momentum) * self.running_mean
                running_deconv = self.momentum * deconv + (1. - self.momentum) * self.running_deconv

                # track stats for evaluation
                self.running_mean.assign(running_mean)
                self.running_deconv.assign(running_deconv)

        else:
            X_mean = self.running_mean
            deconv = self.running_deconv

        # 4. X * deconv * conv = X * (deconv * conv)
        if self.groups == 1:
            w = tf.reshape(self.kernel, [C // block, self.num_features, -1])
            w = tf.transpose(w, [0, 2, 1])
            w = tf.reshape(w, [-1, self.num_features])
            w = tf.matmul(w, deconv)

            if self.use_bias:
                b_dash = tf.matmul(w, (tf.expand_dims(X_mean, axis=-1)))
                b_dash = tf.reshape(b_dash, [self.filters, -1])
                b_dash = tf.reduce_sum(b_dash, axis=1)
                b = self.bias - b_dash
            else:
                b = 0.

            w = tf.reshape(w, [C // block, -1, self.num_features])
            w = tf.transpose(w, [0, 2, 1])

        else:
            w = tf.reshape(self.kernel, [C // block, -1, self.num_features])
            w = tf.matmul(w, deconv)

            if self.use_bias:
                b_dash = tf.matmul(w, tf.reshape(X_mean, [-1, self.num_features, 1]))
                b_dash = tf.reshape(b_dash, self.bias.shape)
                b = self.bias - b_dash
            else:
                b = 0.

        w = tf.reshape(w, self.kernel.shape)

        x = tf.nn.conv2d(x, w, self.strides, str(self.padding).upper(), dilations=self.dilation_rate)
        if self.use_bias:
            x = tf.nn.bias_add(x, b, data_format="NHWC")

        if self.activation is not None:
            return self.activation(x)
        else:
            return x





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


def UNet_ConvUnit(input_tensor, stage, nb_filter, kernel_size=3, mode='None'):   
    x = FastDeconv2D(nb_filter, kernel_size=(3,3),activation='selu',  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = FastDeconv2D(nb_filter, kernel_size=(3,3),activation='selu',  kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization(name='bn' + stage)(x)

    if mode == 'residual':
        x = Add(name='resi' + stage)([x, x0])
    return x


def EF_UNet(input_shape, classes=1, loss='bce'):
    mode = 'None'
    nb_filter = [32, 64, 128, 256, 512]
    bn_axis = 3
    
    # Left side of the U-Net
    inputs = Input(shape=input_shape, name='input') #batchsize,128,128,26#
    

    conv1 = UNet_ConvUnit(inputs, stage='1', nb_filter=nb_filter[0], mode=mode)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = UNet_ConvUnit(pool1, stage='2', nb_filter=nb_filter[1], mode=mode)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = UNet_ConvUnit(pool2, stage='3', nb_filter=nb_filter[2], mode=mode)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = UNet_ConvUnit(pool3, stage='4', nb_filter=nb_filter[3], mode=mode)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Bottom of the U-Net
    conv5 = UNet_ConvUnit(pool4, stage='5', nb_filter=nb_filter[4], mode=mode)
    
    # Right side of the U-Net
    up1 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up1', padding='same')(conv5)
    merge1 = concatenate([conv4,up1], axis=bn_axis)
    conv6 = UNet_ConvUnit(merge1, stage='6', nb_filter=nb_filter[3], mode=mode)

    up2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up2', padding='same')(conv6)
    merge2 = concatenate([conv3,up2], axis=bn_axis)
    conv7 = UNet_ConvUnit(merge2, stage='7', nb_filter=nb_filter[2], mode=mode)

    up3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up3', padding='same')(conv7)
    merge3 = concatenate([conv2,up3], axis=bn_axis)
    conv8 = UNet_ConvUnit(merge3, stage='8', nb_filter=nb_filter[1], mode=mode)
    
    up4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up4', padding='same')(conv8)
    merge4 = concatenate([conv1,up4], axis=bn_axis)
    conv9 = UNet_ConvUnit(merge4, stage='9', nb_filter=nb_filter[0], mode=mode)

    # Output layer of the U-Net with a softmax activation
    output = Conv2D(classes, (1, 1), activation='sigmoid', name='output', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv9)

    model = Model(inputs, output)

    if loss == 'bce':
        loss = 'binary_crossentropy'
    elif loss == 'wbce':
        loss = weighted_bce_dice_loss

    model.compile(optimizer=Adam(lr=1e-4), loss = loss, metrics = ['accuracy'])    
    
    return model
