import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import pandas as pd

from data import build_sources_from_metadata
from data import make_dataset
import matplotlib.pyplot as plt
import numpy as numpy

# LINEAR MODEL

def LinearModel(num_class, input_shape=(224,224,3)):
    model = models.Sequential()

    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(num_class,activation='softmax'))
    
    return model

# VGG 16

def VGG_16(num_classes,input_shape=(224,224,3)):
    model = models.Sequential()

    model.add(layers.Conv2D(64, input_shape=input_shape, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.Conv2D(64,activation='relu',kernel_size=3,padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2))
    
    model.add(layers.Conv2D(128,activation='relu', kernel_size=3, padding='same'))
    model.add(layers.Conv2D(128, activation='relu', kernel_size=3, padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Conv2D(256,activation='relu', kernel_size=3, padding='same'))
    model.add(layers.Conv2D(256, activation='relu', kernel_size=3, padding='same'))
    model.add(layers.Conv2D(256, activation='relu', kernel_size=3, padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Conv2D(512,activation='relu', kernel_size=3, padding='same'))
    model.add(layers.Conv2D(512, activation='relu', kernel_size=3, padding='same'))
    model.add(layers.Conv2D(512, activation='relu', kernel_size=3, padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Conv2D(512,activation='relu',kernel_size=3, padding='same'))
    model.add(layers.Conv2D(512, activation='relu',kernel_size=3, padding='same'))
    model.add(layers.Conv2D(512, activation='relu',kernel_size=3, padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(4096,activation='relu',))
    model.add(layers.Dense(4096,activation='relu'))
    model.add(layers.Dense(num_classes,activation='softmax'))
    
    return model

# LENET

def LeNet(num_classes,input_shape=(32,32,3)):
    model = models.Sequential()

    model.add(layers.Conv2D(6, input_shape=input_shape, kernel_size=5, activation='tanh'))
    model.add(layers.AvgPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Conv2D(16, activation='tanh',kernel_size=5))
    model.add(layers.AvgPool2D(pool_size=(2, 2),strides=2))

    model.add(layers.Flatten())
    model.add(layers.Dense(120,activation='tanh',))
    model.add(layers.Dense(84, activation='tanh'))

    model.add(layers.Dense(num_classes , activation='softmax'))

    return model


# INCEPTION

def _inception_block(inputs, num_filters=512, activation='relu'):

    x11 = layers.Conv2D(num_filters,activation='relu',kernel_size=1, strides=1, padding='same')(inputs)
    x12 = layers.Conv2D(num_filters,activation='relu',kernel_size=1, strides=1, padding='same')(inputs)
    x13 = layers.Conv2D(num_filters,activation='relu',kernel_size=3, strides=1, padding='same')(inputs)
    x21 = layers.Conv2D(num_filters,activation='relu',kernel_size=1, strides=1, padding='same')(inputs)
    x22 = layers.Conv2D(num_filters,activation='relu',kernel_size=3, strides=1, padding='same')(x11)
    x23 = layers.Conv2D(num_filters,activation='relu',kernel_size=5, strides=1, padding='same')(x12)
    x24 = layers.Conv2D(num_filters,activation='relu',kernel_size=1, strides=1, padding='same')(x13)

    out = tf.keras.layers.Concatenate()([x21, x22, x23, x24])

    return out

def _inception_output(input,num_classes):

    x = layers.AvgPool2D(pool_size=(5,5),strides=1,padding='valid')(input)
    x = layers.Conv2D(128,activation='relu',kernel_size=1, strides=1, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dense(1024,activation='relu',kernel_size=1, strides=1, padding='same')(x)
    x = layers.Dropout(0.7)(x)
    out = layers.Dense(num_classes , activation='softmax')(x)

    return out
 
def inception(num_classes, input_shape, output_layes, maxpool_layers,depth=1):
    # if is isinstance(output_layes, (list,set,tuple)):
    #     output_layes = set(output_layes)
    # else:
    #     raise("Enter a valid output_layers")

    #if depth < len(maxpool_layers) or depth< len(output_layes):
    #    raise("Enter a valid depth") 

    inputs = tf.keras.Input(shape=input_shape, name='img')

    x = layers.Conv2D(64,activation='relu', kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.MaxPool2D(pool_size=(3, 3),strides=2, padding='same')(x)
    x = layers.Conv2D(192,activation='relu',kernel_size=1, strides=1, padding='same')(x)
    x = layers.Conv2D(256,activation='relu',kernel_size=3, strides=1, padding='same')(x)

    x = layers.MaxPool2D(pool_size=(3, 3),strides=2, padding='same')(x)
    x = _inception_block(x, num_filters=512, activation='relu')
    
    x = _inception_block(x, num_filters=512, activation='relu')

    x = layers.MaxPool2D(pool_size=(3, 3),strides=2, padding='same')(x)
    x = _inception_block(x, num_filters=512, activation='relu')

    out1 = _inception_output(x,num_classes)
    x = _inception_block(x, num_filters=512, activation='relu')

    x = _inception_block(x, num_filters=512, activation='relu')

    x = _inception_block(x, num_filters=512, activation='relu')

    out2 = _inception_output(x,num_classes)
    x = _inception_block(x, num_filters=512, activation='relu')

    x = layers.MaxPool2D(pool_size=(3, 3),strides=2, padding='same')(x)
    x = _inception_block(x, num_filters=512, activation='relu')

    x = _inception_block(x, num_filters=512, activation='relu')

    x = layers.AvgPool2D(pool_size=(7,7),strides=1,padding='valid')(x)
    x = tf.keras.layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',kernel_size=1, strides=1, padding='same')(x)
    out3 = layers.Dense(num_classes , activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[out1, out2, out3], name='inception')
    return model

# RES NET

def _conv_batch_act(inputs, filters, kernel_size, padding='same', strides=(1, 1), activation='relu'):
    x = tf.keras.layers.Conv2D(filters,
           kernel_size=kernel_size, padding=padding, strides=strides)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x
def ResNet(stages, num_classes, activation):
    """
    This function implements the ResNet architecture. This function is
    rensponsible for mapping a batch of RGB images to the output of
    the last leyer of the neural network.
    
    Args:
        inputs (tf.Tensor): a batch of RGB images in NHWC format.
        stages (tuple or list): Number of residual block per stage
        num_classes (int): number of classes of the classification problem,
            which corresponds to the output of the last layer of the
            neural network.
            
    Returns:
        (tf.Tensor): The output of the last layer of the
            ResNet model. It must have shape (N, num_classes).
    """
    inputs = tf.keras.Input(shape=(224, 224, 3), name='img')
    
    # First convolution
    x = _conv_batch_act(inputs, 64, (5, 5), padding='same', strides=(1, 1), activation=activation)
    
    input_filters = 64
    filters = (64, 256)
    downsample = False
    # RestNet structure
    for num_blocks in stages:
        x = _residual_block_stage(inputs=x, num_blocks=num_blocks,
            input_filters=input_filters, filters=filters,
            downsample=downsample, activation=activation
        )
        
        downsample = True
        input_filters = filters[1]
        filters = (filters[0]*2, filters[1]*2)
        
    x = layers.GlobalAveragePooling()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x, name='resnet')
    return model
def _residual_block(inputs, input_filters, filters,
        downsample=False, activation='relu', **kwargs):
    
    x = _conv_batch_act(inputs, filters[0], (1, 1), activation)
    
    if downsample:
        s = 2
    else:
        s = 1
        
    x = _conv_batch_act(inputs, filters[0], (3, 3), strides=(s, s), activation=activation)
    x = _conv_batch_act(inputs, filters[1], (1, 1), activation)
    
    if downsample:
        x2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(3, 3),
                strides=(1, 1), padding='same')(x)
    elif filters[1] != input_filters:
        x2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(1, 1),
                strides=(1, 1), padding='same')(x)
    else:
        x2 = inputs
        
    x = tf.keras.layers.Add()([x, x2])
    x = tf.keras.layers.Activation(activation)(x)
    return x

def _residual_block_stage(inputs, num_blocks, input_filters, filters, activation, downsample=False):

    feature_map = inputs
    previous_filters = input_filters
    downsample_ = downsample
 
    for i in range(num_blocks):

        if (downsample and i == 0):
            downsample_ = True

        feature_map = _residual_block(feature_map, previous_filters,
            filters, downsample_, activation)
        previous_filters = tf.shape(feature_map)[-1]

        downsample_ = False

    return feature_map

if __name__ == '__main__':
    pass


    