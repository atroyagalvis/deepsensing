"""this file contains functions for constructing various keras models"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers import Reshape, Flatten, Input, Dense, Conv2D, MaxPooling2D, Dropout
from keras.models import Model



def VGG16_like_fine_tuning(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #create the VGG16 model
  #Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=False)(input_img)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block5_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu', name="vgg_dense1")(encoded)
  output = Dropout(0.5)(output)
  ouput = Dense(img_rows*img_cols,activation='relu', name="vgg_dense2")(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


"""
Build a model inspired from the VGG16 model but generalized to work with multi-spectral images 
img_bands : number of spectral bands of each image
img_rows : number of rows in each image tile
img_cols : number of columns in each image tile
"""
def VGG16_like(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #create the VGG16 model
  #Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block5_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu',name="vgg_dense1")(encoded)
  output = Dropout(0.5)(output)
  ouput = Dense(img_rows*img_cols,activation='relu',name="vgg_dense2")(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def simple_autoencoder(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols*img_bands,activation='linear', name="denseauto")(encoded)
  output = Reshape((img_bands,img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn_fine_tuning(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=False)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2' , trainable=False)(input_img)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn")(encoded)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn2")(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv2' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn")(encoded)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn2")(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn_categorical(img_bands = 4, img_rows = 64, img_cols = 64, nb_classes = 2):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(1024,activation='relu', name="dense_basic_cat")(encoded)
  output = Dense(nb_classes,activation='softmax', name="output_basic_cat")(encoded)  
  return Model(input_img, output)  

def VGG16_like_fewfilters(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #create the VGG16 model
  #Block 1
  x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)

  # Block 2
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)

  # Block 3
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block3_pool')(x)

  # Block 4
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block4_pool')(x)

  # Block 5
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block5_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='linear')(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn2(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  
  #Block 1
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', name='block1_pool')(x)
  # Block 2
  x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', name='block2_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu')(encoded)
  output = Dropout(0.5)(output)
  ouput = Dense(img_rows*img_cols,activation='linear')(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)
  

