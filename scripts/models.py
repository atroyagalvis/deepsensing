"""this file contains functions for constructing various keras models"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.layers import Reshape, Flatten, Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation 
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras


def VGG16_like_fine_tuning(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #create the VGG16 model
  #Block 1
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=False)(input_img)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=False)(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=False)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv2', trainable=False)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv1', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv2', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv1', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv2', trainable=False)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv3', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block5_pool')(x)
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
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)

  # Block 2
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)

  # Block 3
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=True)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv2', trainable=True)(x)
  x = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)

  # Block 4
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv1', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv2', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block4_pool')(x)

  # Block 5
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv1', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv2', trainable=True)(x)
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block5_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu',name="vgg_dense1")(encoded)
  output = Dropout(0.5)(output)
  ouput = Dense(img_rows*img_cols,activation='relu',name="vgg_dense2")(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def simple_autoencoder(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv2' , trainable=True)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv3' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=True)(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols*img_bands,activation='linear', name="denseauto")(encoded)
  output = Reshape((img_bands,img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn_fine_tuning(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=False)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv2' , trainable=False)(input_img)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv3' , trainable=False)(input_img)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=False)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=False)(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv2', trainable=False)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='new_conv', trainable=True)(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn")(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn")(encoded)
  output = Dense(img_rows*img_cols,activation='relu', name="dense_basic_cnn2")(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn_categorical(img_bands = 4, img_rows = 64, img_cols = 64, nb_classes = 2):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #Block 1
  x = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)
  # Block 2
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)
  # Block 3
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)
  encoded = Flatten()(x)
  output = Dense(1024,activation='relu', name="dense_basic_cat")(encoded)
  output = Dense(nb_classes,activation='softmax', name="output_basic_cat")(output)  
  return Model(input_img, output)  

def VGG16_like_fewfilters(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  #create the VGG16 model
  #Block 1
  x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = Conv2D(16, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)

  # Block 2
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv2', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)

  # Block 3
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block3_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block3_pool')(x)

  # Block 4
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block4_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block4_pool')(x)

  # Block 5
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv1', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv2', trainable=True)(x)
  x = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block5_conv3', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block5_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='linear')(encoded)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)


def basic_cnn2(img_bands = 4, img_rows = 64, img_cols = 64):
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  
  #Block 1
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block1_conv1' , trainable=True)(input_img)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block1_pool')(x)
  # Block 2
  x = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_first", name='block2_conv1', trainable=True)(x)
  x = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block2_pool')(x)
  encoded = Flatten()(x)
  output = Dense(img_rows*img_cols,activation='relu')(encoded)
  output = Dropout(0.5)(output)
  ouput = Dense(img_rows*img_cols,activation='linear')(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)
  
def residual_block(img_rows = 64, img_cols = 64, in_filters=64, out_filters=64, name='resblock'):
  
  x = Input(shape=(in_filters, img_rows, img_cols))
  bottleneck_filters = in_filters/4

  #1x1 convolution - reduce dimensionality 
  y = BatchNormalization(axis=1)(x)
  
  y = Activation('relu')(y)
  
  y = Conv2D(bottleneck_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_1')(y)
  print Model(x,y).output_shape
  #3x3 convolution
  y = BatchNormalization()(y)
  y = Activation('relu')(y)
  y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_1')(y)
  
  #3x3 convolution
  y = BatchNormalization()(y)
  y = Activation('relu')(y)
  y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_2')(y)
  
  #1x1 convolution - restore dimensionality
  y = BatchNormalization()(y)
  y = Activation('relu')(y)
  y = Conv2D(out_filters,(1,1), activation='relu', padding='same', data_format="channels_first", name=name+'conv1x1_2')(y)
  print Model(x,y).output_shape
  return keras.layers.add([x, y])

"""defines a residual network model
nb_blocs : number of blocks of reducing image size and increasing filter depth
res_blocks : number of residual block for each main block, if len(res_blocks)==1 the same number of residual blocks is added for every block; otherwise len(res_blocks) should be equal to nb_blocs and each element of the list correspond to the number of residual blocks in the corresponding main block
filter_depth : list of the number of filters for each main block len(filter_depth) have to be equal to nb_blocks
"""
def residual_net(img_bands = 4, img_rows = 64, img_cols = 64,nb_blocks=4, res_blocks=[3,4,6,3], filter_depth=[64,128,256,512]):
  assert len(filter_depth) == nb_blocks, "filter_depth should have nb_blocks elements"
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  res_net = Conv2D(64, (7, 7), activation='relu', padding='same', data_format="channels_first", name='resnet_first_conv', trainable=True)(input_img)
  res_net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='pool')(res_net)
  for b in xrange(nb_blocks):
    if len(res_blocks) == 1:#same number of residual blocks
      for rb in xrange(res_blocks[0]):
        if rb == 0: #do pooling
          res_net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='resblock_pool'+str(b))(res_net)
        res_net = residual_block(img_rows, img_cols, in_filters=filter_depth[b], out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)
    else:
      assert len(res_blocks)==nb_blocks, "res_blocks should have either 1 or nb_blocks elements"
      for rb in xrange(res_blocks[b]):
        if rb == 0: #do pooling
          res_net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='resblock_pool'+str(b))(res_net)      
        res_net = residual_block(img_rows, img_cols, in_filters=filter_depth[b], out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)
  output = Flatten()(res_net)
  output = Dense(img_rows*img_cols,activation='linear')(output)
  output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)  



















