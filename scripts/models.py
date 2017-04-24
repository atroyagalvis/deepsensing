"""this file contains functions for constructing various keras models"""
import numpy as np
#np.random.seed(1337)  # for reproducibility
from keras.layers import Reshape, Flatten, Input, Dense, Conv2D, MaxPooling2D, Dropout, Activation, Merge
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import initializers
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



def VGG16_like(img_bands = 4, img_rows = 64, img_cols = 64):
  """
Build a model inspired from the VGG16 model but generalized to work with multi-spectral images 
img_bands : number of spectral bands of each image
img_rows : number of rows in each image tile
img_cols : number of columns in each image tile
"""
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
  if in_filters != out_filters:
    xadd =  Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", strides=2, name=name+'dim_change')(x)
    #1x1 convolution - reduce dimensionality 
    y = Conv2D(bottleneck_filters,(1,1), padding='same', data_format="channels_first", strides=2, name=name+'conv1x1_1')(x)
    #3x3 convolution
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
    #3x3 convolution
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_2',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
    #1x1 convolution restore dimensionality
    y = Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_2')(y)
  else:
    xadd = x
    #1x1 convolution - reduce dimensionality 
    y = Conv2D(bottleneck_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_1')(x)
    #3x3 convolution
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
    #3x3 convolution
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_2',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01))(y)
    #1x1 convolution - restore dimensionality
    y = Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_2')(y)
  z = keras.layers.add([xadd, y])
  return Model(x,z)

def residual_block_post_activation(img_rows = 64, img_cols = 64, in_filters=64, out_filters=64, name='resblock', initializer='random_uniform'):
  x = Input(shape=(in_filters, img_rows, img_cols))
  bottleneck_filters = in_filters/4
  if in_filters != out_filters:
    xadd =  Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", strides=2, name=name+'dim_change',kernel_initializer=initializer)(x)
    #1x1 convolution - reduce dimensionality 
    y = Conv2D(bottleneck_filters,(1,1), padding='same', data_format="channels_first", strides=2, name=name+'conv1x1_1',kernel_initializer=initializer)(x)
    #3x3 convolution
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer=initializer)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    #3x3 convolution
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_2',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer=initializer)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    #1x1 convolution restore dimensionality
    y = Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_2',kernel_initializer=initializer)(y)
  else:
    xadd = x
    #1x1 convolution - reduce dimensionality 
    y = Conv2D(bottleneck_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_1',kernel_initializer=initializer)(x)
    #3x3 convolution
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_1',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer=initializer)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    #3x3 convolution
    y = Conv2D(bottleneck_filters,(3,3),  padding='same', data_format="channels_first", name=name+'conv3x3_2',kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l2(0.01),kernel_initializer=initializer)(y)
    y = BatchNormalization(axis=1)(y)
    y = Activation('relu')(y)
    #1x1 convolution - restore dimensionality
    y = Conv2D(out_filters,(1,1), padding='same', data_format="channels_first", name=name+'conv1x1_2',kernel_initializer=initializer)(y)
  z = keras.layers.add([xadd, y])
  return Model(x,z)


def residual_net(img_bands = 4, img_rows = 64, img_cols = 64,nb_blocks=4, res_blocks=[3,4,6,3], filter_depth=[64,128,256,512], categorical=False,nb_classes=-1):
  """defines a residual network model
nb_blocs : number of blocks of reducing image size and increasing filter depth
res_blocks : number of residual block for each main block, if len(res_blocks)==1 the same number of residual blocks is added for every block; otherwise len(res_blocks) should be equal to nb_blocs and each element of the list correspond to the number of residual blocks in the corresponding main block
filter_depth : list of the number of filters for each main block len(filter_depth) have to be equal to nb_blocks
"""
  assert len(filter_depth) == nb_blocks, "filter_depth should have nb_blocks elements"
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  current_depth = 64 
  res_net = Conv2D(current_depth, (7, 7), activation='relu', padding='same', strides=2, data_format="channels_first", name='resnet_first_conv', trainable=True)(input_img)
  res_net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='pool')(res_net)
  for b in xrange(nb_blocks):
    if len(res_blocks) == 1:#same number of residual blocks
      for rb in xrange(res_blocks[0]):
        if rb == res_blocks[0]-1 and b != nb_blocks-1:
          res_net = residual_block(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b+1], name='resblock'+str(rb))(res_net)
        else:
          res_net = residual_block(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)          
    else:
      assert len(res_blocks)==nb_blocks, "res_blocks should have either 1 or nb_blocks elements"
      for rb in xrange(res_blocks[b]):
        if rb == res_blocks[b]-1 and b != nb_blocks-1: 
          res_net = residual_block(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b+1], name='resblock'+str(rb))(res_net)
        else:
          res_net = residual_block(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)
      a,current_depth,b,c = Model(input_img,res_net).output_shape 
    
  output = Flatten()(res_net)
  if categorical:
    assert nb_classes != -1, 'parameter nb_classes should be defined'
    output = Dense(1024,activation='relu', name="dense_resnet_cat")(output)
    droprate = 1-(min(1024,nb_classes*10)/1024.)
    output = Dropout(droprate)(output)
    output = Dense(nb_classes,activation='softmax', name="output_resnet_cat")(output)
  else:
    output = Dense(img_rows*img_cols,activation='linear',name="output_resnet")(output)
    output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)  

def residual_net_post_activation(img_bands = 4, img_rows = 64, img_cols = 64,nb_blocks=4, res_blocks=[3,4,6,3], filter_depth=[64,128,256,512], categorical=False,nb_classes=-1, initializer='random_uniform'):
  """defines a residual network model
nb_blocs : number of blocks of reducing image size and increasing filter depth
res_blocks : number of residual block for each main block, if len(res_blocks)==1 the same number of residual blocks is added for every block; otherwise len(res_blocks) should be equal to nb_blocs and each element of the list correspond to the number of residual blocks in the corresponding main block
filter_depth : list of the number of filters for each main block len(filter_depth) have to be equal to nb_blocks
"""
  assert len(filter_depth) == nb_blocks, "filter_depth should have nb_blocks elements"
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  current_depth = 64 
  res_net = Conv2D(current_depth, (7, 7), activation='relu', padding='same', strides=2, data_format="channels_first", name='resnet_first_conv', trainable=True,kernel_initializer=initializer)(input_img)
  res_net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='pool')(res_net)
  for b in xrange(nb_blocks):
    if len(res_blocks) == 1:#same number of residual blocks
      for rb in xrange(res_blocks[0]):
        if rb == res_blocks[0]-1 and b != nb_blocks-1:
          res_net = residual_block_post_activation(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b+1], name='resblock'+str(rb))(res_net)
        else:
          res_net = residual_block_post_activation(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)          
    else:
      assert len(res_blocks)==nb_blocks, "res_blocks should have either 1 or nb_blocks elements"
      for rb in xrange(res_blocks[b]):
        if rb == res_blocks[b]-1 and b != nb_blocks-1: 
          res_net = residual_block_post_activation(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b+1], name='resblock'+str(rb))(res_net)
        else:
          res_net = residual_block_post_activation(img_rows, img_cols, in_filters=current_depth, out_filters=filter_depth[b], name='resblock'+str(rb))(res_net)
      a,current_depth,b,c = Model(input_img,res_net).output_shape 
    
  output = Flatten()(res_net)
  if categorical:
    assert nb_classes != -1, 'parameter nb_classes should be defined'
    output = Dense(1024,activation='relu', name="dense_resnet_cat")(output)
    droprate = 1-(min(1024,nb_classes*10)/1024.)
    output = Dropout(droprate)(output)
    output = Dense(nb_classes,activation='softmax', name="output_resnet_cat")(output)
  else:
    output = Dense(img_rows*img_cols,activation='linear',name="output_resnet")(output)
    output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)    

def conv_pool_dense_net(img_bands = 4, img_rows = 64, img_cols = 64,nb_blocks=4, in_blocks=[3,4,6,3], filter_depth=[64,128,256,512], dense_layers=1 ,categorical=False,nb_classes=-1, droprate=0.9):
  """
  Defines a classical convolution-pooling-dense network
"""
  assert len(filter_depth) == nb_blocks, "filter_depth should have nb_blocks elements"
  input_img = Input(shape=(img_bands, img_rows, img_cols))
  net = input_img
  for b in xrange(nb_blocks):
   
    if len(in_blocks) == 1:#same number of residual blocks
      blocks = in_blocks[0]
    else:
      assert len(in_blocks)==nb_blocks, "res_blocks should have either 1 or nb_blocks elements"
      blocks = in_blocks[b]   
    
    for rb in xrange(blocks):
      net = Conv2D(filter_depth[b], (3, 3), activation='relu', padding='same', data_format="channels_first", name='block'+str(b)+'_conv'+str(rb) , trainable=True)(net)
      net = Dropout(0.1)(net)
    net = MaxPooling2D((2, 2), padding='same', data_format="channels_first", name='block'+str(b)+'_pool')(net)    
  output = Flatten()(net)
  for d in xrange(dense_layers):
    output = Dense(1024,activation='relu', name="dense_layer"+str(d))(output)
    output = Dropout(droprate)(output)
  if categorical:
    assert nb_classes != -1, 'parameter nb_classes should be defined'  
    output = Dense(nb_classes,activation='softmax', name="output_layer")(output)
  else:
    output = Dense(img_rows*img_cols,activation='linear',name="output_resnet")(output)
    output = Reshape((img_rows,img_cols))(output)
  return Model(input_img, output)  
















