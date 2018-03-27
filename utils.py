"""This file provides many miscelaneous functions to load and process GeoTIFF images in order to provide training and testing sets for deep learning with keras"""
import gdal
import numpy as np
np.random.seed(1337)  # for reproducibility
import threading
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from  keras import backend as K
import tensorflow as tf

def standardize_image(a):
  """
Standardize an image by removing the mean of each band and then dividing by the standard deviation of the band
a : the image to be standardized encoded as a numpy array
t : the image from which the pre-processing statistics are computed
"""
  a = a.astype('float32')
  for i in range(len(a)):
    a[i] -= a[i].mean()
    a[i] /= a[i].std()
  return a

def zca_whitening(x):
  """
Applies ZCA whitening to an image
a : the image to process encoded as a numpy array
"""
  flat_x = np.reshape(x, (x.size,1))
  sigma = np.dot(flat_x.T,flat_x) / flat_x.shape[0]
  u, s, _ = np.linalg.svd(sigma)
  principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)
  whitex = np.dot(flat_x, principal_components)
  x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
  return x


class batchGenerator:
    """
generator to produce training image tile couples (image tile, labels) 
imagefile : string containing the path to the raw image to load
vtfile : string containing the path to the target labels, if no parameter passed, the generator will return the same input as target (autoencoder)
tilesize : the size of each generated tile (tiles are always generated as square matrices)
NOTE: imagefile and vtfile should correspond to the exact same area otherwise the training will surely fail
"""
    def __init__(self, imagefile, vtfile='autoencode', tilesize=16, batch_size=32, random=True, categorical=False, nb_classes=-1, data_augmentation=True, target_class=-1, collaborative = False, softmax=False):
        #if os.path.isdir(imagefile):#if file is a folder we load every pair of image/label image files in the folder        
         # files = os.listdir(imagefile)
         # for f in files:
         #   if os.path.isfile(f):
         #     data = gdal.Open(f)
        print( "generating batches on image", imagefile)
        self.data = gdal.Open(imagefile)
        print(self.data)
        self.random = random
        self.categorical = categorical
        self.nb_classes = nb_classes
        self.data_augmentation = data_augmentation
        self.xsize = self.data.RasterXSize
        self.ysize = self.data.RasterYSize
        #
        print("preprocessing image")
        self.image = standardize_image(self.data.ReadAsArray())
        #self.image = zca_whitening(self.image)
        if vtfile is not 'autoencode':
          self.autoencode = False 
          self.imagevt = gdal.Open(vtfile).ReadAsArray()
        else:
          self.autoencode = True  
        self.tilesize = tilesize
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.current_index = 0
        self.target_class = target_class
        self.collaborative = collaborative
        self.softmax = softmax
        if -1 in self.imagevt:
          self.possible_coords = np.where(self.imagevt != -1)
        else:
          self.possible_coords = None

    def __iter__(self):
        return self

    def __next__(self):       
        with self.lock:
          X = []
          y = []
          for b in range(self.batch_size):
            if self.random:
              if self.possible_coords is not None:
                rx = 0
                ry = 0
                while rx <= self.tilesize or rx >= self.xsize-self.tilesize:   
                  rx = self.possible_coords[1][np.random.randint(len(self.possible_coords[1]))]
                while ry <= self.tilesize or ry >= self.ysize-self.tilesize:   
                  ry = self.possible_coords[0][np.random.randint(len(self.possible_coords[0]))]
              else:                
                rx = np.random.randint(self.xsize-self.tilesize)
                ry = np.random.randint(self.ysize-self.tilesize)    
            else:
              rx = ((self.current_index+b)*self.tilesize)%(self.xsize-self.tilesize)
              ry = ((((self.current_index+b)*self.tilesize)/(self.xsize))*self.tilesize)%(self.ysize-self.tilesize)
              
            sample = self.image[:,ry:ry+self.tilesize,rx:rx+self.tilesize]
            
            if self.target_class == -1:#we are trying multiclass classification
              label_img = self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize].reshape((1,self.tilesize,self.tilesize)) 
            else:#monoclass classification
              label_img = (self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize]==self.target_class).astype(np.int) 

            if self.data_augmentation:
              lbl = np.asarray(label_img)
              assert len(sample.shape) == len(lbl.shape), 'bad shapes'
              sample, label_img = data_augmentation(sample,lbl)

            if self.autoencode:
              labels = sample
            elif self.categorical:
              assert self.nb_classes != -1, 'parameter nb_classes should be defined'
              labels = np.zeros(self.nb_classes) 
              #we take the central class as the representing class
              labels[int(label_img[self.tilesize/2,self.tilesize/2])] = 1              
            else:
              if self.collaborative:
                class_channels = []
                assert self.nb_classes != -1, "nb_classes has to be specified for the collaborative model"
                for c in range(self.nb_classes):
                  class_channels.append((label_img==c).astype(np.int))
                if self.softmax:
                  labels = np.asarray(class_channels).reshape(1,self.nb_classes,self.tilesize,self.tilesize)
                else:
                  class_channels.append(np.asarray(label_img))
                  labels = np.asarray(class_channels)              
              else:
                labels = label_img.reshape((1,self.tilesize,self.tilesize))
                 
          #  write_array_as_tif(sample, 'test'+str(b)+str(np.random.randint(100))+'.tif', trans = self.data.GetGeoTransform(), proj = self.data.GetProjection(), xoff=rx, yoff=ry )     
            X.append(sample)
            y.append(labels) 
            
          self.current_index += self.batch_size 
          X = np.asarray(X)
          y = np.asarray(y)
          if self.collaborative:
            y = np.rollaxis(y,1)
            y = list(y)
       
          
          return X,y


class batchGeneratorFolder(object):
    """
generator to produce training image tile couples (image tile, labels) 
imagefile : string containing the path to the raw image to load
vtfile : string containing the path to the target labels, if no parameter passed, the generator will return the same input as target (autoencoder)
tilesize : the size of each generated tile (tiles are always generated as square matrices)
NOTE: imagefile and vtfile should correspond to the exact same area otherwise the training will surely fail
"""
    def __init__(self, folder, validation_img, tilesize=16, batch_size=32, random=True, categorical=False, nb_classes=-1, data_augmentation=True, target_class=-1, collaborative = False, softmax=False):
        #if os.path.isdir(imagefile):#if file is a folder we load every pair of image/label image files in the folder        
         # files = os.listdir(imagefile)
         # for f in files:
         #   if os.path.isfile(f):
         #     data = gdal.Open(f)
        print( "training on folder", folder)
        print( "validation image", validation_img)
        self.training_images = []
        self.training_labels = []
        self.imagenames = []
        assert os.path.isdir(folder), "you must specify a valid folder"
        dirList = os.listdir(folder) # current directory
        for dir in dirList:
          if dir.endswith(".tif") and dir not in validation_img:
            if "labels" not in dir:
              assert os.path.isfile(folder+"labels"+dir), folder+"labels"+dir +" does not exist"
              self.imagenames.append(dir)
              self.training_labels.append(gdal.Open(folder+"labels"+dir).ReadAsArray()) #label images should all have the same name as the raw image and prefix "labels"
              self.training_images.append(standardize_image(gdal.Open(folder+dir).ReadAsArray()))
        print(self.imagenames)
        self.random = random
        self.categorical = categorical
        self.nb_classes = nb_classes
        self.data_augmentation = data_augmentation
        self.tilesize = tilesize
        self.batch_size = batch_size
        self.lock = threading.Lock()
        self.current_index = 0
        self.target_class = target_class
        self.collaborative = collaborative
        self.softmax = softmax
        print("kaka")

    def __iter__(self):
        return self

    def __next__(self):       
        with self.lock:
          X = []
          y = []
          for b in range(self.batch_size):
            imageindex = np.random.randint(len(self.training_images))
            self.image = self.training_images[imageindex]
            self.imagevt = self.training_labels[imageindex]
            
            #if -1 in self.imagevt:
             # print "labels should be >=0"
              #exit()

            #assert self.image[0].shape == self.imagevt.shape, "raw image and labels image should have the same dimensions "+self.imagenames[imageindex]
            _,self.ysize, self.xsize = self.image.shape
     
            if self.random:
              rx = np.random.randint(self.xsize-self.tilesize)
              ry = np.random.randint(self.ysize-self.tilesize)    
            else:
              rx = ((self.current_index+b)*self.tilesize)%(self.xsize-self.tilesize)
              ry = ((((self.current_index+b)*self.tilesize)/(self.xsize))*self.tilesize)%(self.ysize-self.tilesize)
              
            sample = self.image[:,ry:ry+self.tilesize,rx:rx+self.tilesize]
            
            if self.target_class == -1:#we are trying multiclass classification
              label_img = self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize].reshape((1,self.tilesize,self.tilesize)) 
            else:#monoclass classification
              label_img = (self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize]==self.target_class).astype(np.int) 

            if self.data_augmentation:
              lbl = np.asarray(label_img)
              #assert len(sample.shape) == len(lbl.shape), 'bad shapes'
              sample, label_img = data_augmentation(sample,lbl)

            if self.categorical:
              #assert self.nb_classes != -1, 'parameter nb_classes should be defined'
              labels = np.zeros(self.nb_classes) 
              #we take the central class as the representing class
              labels[int(label_img[self.tilesize/2,self.tilesize/2])] = 1              
            else:
              if self.collaborative:
                class_channels = []
                #assert self.nb_classes != -1, "nb_classes has to be specified for the collaborative model"
                for c in range(self.nb_classes):
                  class_channels.append((label_img==c).astype(np.int))
                if self.softmax:
                  labels = np.asarray(class_channels).reshape(1,self.nb_classes,self.tilesize,self.tilesize)
                else:
                  class_channels.append(np.asarray(label_img))
                  labels = np.asarray(class_channels)              
              else:
                labels = label_img.reshape((1,self.tilesize,self.tilesize))
                 
          #  write_array_as_tif(sample, 'test'+str(b)+str(np.random.randint(100))+'.tif', trans = self.data.GetGeoTransform(), proj = self.data.GetProjection(), xoff=rx, yoff=ry )     
            X.append(sample)
            y.append(labels) 
            
          self.current_index += self.batch_size 
          X = np.asarray(X)
          y = np.asarray(y)
          if self.collaborative:
            y = np.rollaxis(y,1)
            y = list(y)
       
          
          return X,y


def data_augmentation(raw, labels):
  """
perform random transformations on the raw and target images
concatenates the raw image with the target labelles image to apply the 
same random transformation to both images
return the transformed images separatedly
"""
  transform_tensor = np.concatenate((raw,labels))  
  data_augmenter = ImageDataGenerator(rotation_range=180.,
                 width_shift_range=0.2,
                 height_shift_range=0.2,
                 zoom_range=0.9,
                 shear_range = 0.785398, 
                 fill_mode='nearest',
                 horizontal_flip=True,
                 vertical_flip=True,
                 data_format= 'channels_first')
  transform_tensor = data_augmenter.random_transform(transform_tensor)
  raw = transform_tensor[0:len(transform_tensor)-1,:,:]
  labels = transform_tensor[len(transform_tensor)-1,:,:]
  return raw,labels



def predict_image_tile_new(model,imagefile,tilesize=16,outputimg = "prediction.tif"):
  """
Makes predictions given a model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training)
outputimg : the target path in which the predictions are stored
"""  
  print("Predicting image "+imagefile)
  data = gdal.Open(imagefile)
  xsize = data.RasterXSize
  ysize = data.RasterYSize
  if(xsize%tilesize == 0):
    xchunks = xsize/tilesize
  else:
    xchunks = (xsize/tilesize)+1
  
  if(ysize%tilesize == 0):
    ychunks = ysize/tilesize
  else:
    ychunks = (ysize/tilesize)+1

  image = standardize_image(data.ReadAsArray())
  #image = zca_whitening(self.data.ReadAsArray())
  #image = zca_whitening(image)
  #image = standardize_image(image)
  
  trans       = data.GetGeoTransform()
  proj        = data.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)
  
  for i in range(xchunks):
    for j in range(ychunks):
      X = np.asarray([ image[:,j*tilesize:j*tilesize+tilesize,i*tilesize:i*tilesize+tilesize] ])
      _,_,w,h = X.shape
      if h != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,0),(0,tilesize-h)), mode='constant')
      if w != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,tilesize-w),(0,0)), mode='constant')
      pred = model.predict(X,verbose=0)
      band = outdata.GetRasterBand(1)
      xoff = i*tilesize
      yoff = j*tilesize
      band.WriteArray(pred[0,0,0:w,0:h], xoff, yoff)
      outdata.FlushCache()
  outdata.FlushCache()


def predict_image_tile_multi(model,imagefile,tilesize=16,outputimg = "prediction.tif", save_all=False):
  """
Makes predictions given a multi output model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training)
outputimg : the target path in which the predictions are stored
"""  
  print("Predicting image "+imagefile)
  print("saving prediction in", outputimg)
  data = gdal.Open(imagefile)
  xsize = data.RasterXSize
  ysize = data.RasterYSize
  if(xsize%tilesize == 0):
    xchunks = xsize/tilesize
  else:
    xchunks = (xsize/tilesize)+1
  
  if(ysize%tilesize == 0):
    ychunks = ysize/tilesize
  else:
    ychunks = (ysize/tilesize)+1

  image = standardize_image(data.ReadAsArray())
  #image = zca_whitening(self.data.ReadAsArray())
  #image = zca_whitening(image)
  #image = standardize_image(image)
  trans       = data.GetGeoTransform()
  proj        = data.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  nb_outputs = len(model.output_shape)
  if save_all:
    outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, nb_outputs, gdal.GDT_Float32)
  else:
    outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)
  
  for i in range(xchunks):
    for j in range(ychunks):
      X = np.asarray([ image[:,j*tilesize:j*tilesize+tilesize,i*tilesize:i*tilesize+tilesize] ])
      _,_,w,h = X.shape
      if h != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,0),(0,tilesize-h)), mode='constant')
      if w != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,tilesize-w),(0,0)), mode='constant')
      pred = model.predict(X,verbose=0)
      if save_all:
        for c in range(nb_outputs): 
          predtile = pred[c]       
          band = outdata.GetRasterBand(c+1)
          xoff = i*tilesize
          yoff = j*tilesize
          band.WriteArray(predtile[0,0,0:w,0:h], xoff, yoff)
      else:
        predtile = pred[-1]       
        band = outdata.GetRasterBand(1)
        xoff = i*tilesize
        yoff = j*tilesize
        band.WriteArray(predtile[0,0,0:w,0:h], xoff, yoff)
      outdata.FlushCache()        
  outdata.FlushCache()

def predict_image_tile_argmax(model,imagefile,tilesize=16,outputimg = "prediction.tif"):
  """
Makes predictions given a multi output model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training)
outputimg : the target path in which the predictions are stored
"""  
  print("Predicting image "+imagefile)
  print("saving prediction in", outputimg)
  data = gdal.Open(imagefile)
  xsize = data.RasterXSize
  ysize = data.RasterYSize
  if(xsize%tilesize == 0):
    xchunks = xsize/tilesize
  else:
    xchunks = (xsize/tilesize)+1
  
  if(ysize%tilesize == 0):
    ychunks = ysize/tilesize
  else:
    ychunks = (ysize/tilesize)+1

  image = standardize_image(data.ReadAsArray())
  #image = zca_whitening(self.data.ReadAsArray())
  #image = zca_whitening(image)
  #image = standardize_image(image)
  trans       = data.GetGeoTransform()
  proj        = data.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  nb_outputs = len(model.output_shape)
  outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)
  
  for i in range(xchunks):
    for j in range(ychunks):
      X = np.asarray([ image[:,j*tilesize:j*tilesize+tilesize,i*tilesize:i*tilesize+tilesize] ])
      _,_,w,h = X.shape
      if h != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,0),(0,tilesize-h)), mode='constant')
      if w != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,tilesize-w),(0,0)), mode='constant')
      pred = model.predict(X,verbose=0)
      predtile = np.argmax(pred, axis=1)
      band = outdata.GetRasterBand(1)
      xoff = i*tilesize
      yoff = j*tilesize
      band.WriteArray(predtile[0,0:w,0:h], xoff, yoff)
      outdata.FlushCache()        
  outdata.FlushCache()


def predict_image_categorical_tile(model,imagefile,tilesize=16,outputimg = "prediction.tif"):
  """
Makes predictions given a model and an raw image file 
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training) each tile is mapped to a single class every pixel of the tile is given the predicted class in the output label map
outputimg : the target path in which the predictions are stored
"""  
  print("Predicting image "+imagefile)
  data = gdal.Open(imagefile)
  xsize = data.RasterXSize
  ysize = data.RasterYSize
  if(xsize%tilesize == 0):
    xchunks = xsize/tilesize
  else:
    xchunks = (xsize/tilesize)+1
  
  if(ysize%tilesize == 0):
    ychunks = ysize/tilesize
  else:
    ychunks = (ysize/tilesize)+1

  image = standardize_image(data.ReadAsArray())
  #image = zca_whitening(image)
  #image = zca_whitening(self.data.ReadAsArray())
  #image = standardize_image(image)
  trans       = data.GetGeoTransform()
  proj        = data.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)

  for i in range(0,xsize,tilesize):
    for j in range(0,ysize,tilesize): 
      X = np.asarray([ image[:,j:j+tilesize,i:i+tilesize]])
      _,_,w,h = X.shape
      if h != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,0),(0,tilesize-h)), mode='constant')
        
      if w != tilesize:
        X = np.pad(X, ((0,0),(0,0),(0,tilesize-w),(0,0)), mode='constant')
        
      labelpatch = np.zeros((w,h))
      pred = model.predict(X,verbose=0)
      label, = np.where(pred[0] == max(pred[0]))     
           
      labelpatch.fill(label[0])      
      band = outdata.GetRasterBand(1)
      
      try:
        band.WriteArray(labelpatch, i, j)
      except:
        print(i,j, labelpatch.shape)
      outdata.FlushCache()
  outdata.FlushCache()



def predict_image_categorical_center(model,imagefile, tilesize=16, centersize=1,outputimg = "prediction.tif", batch_size=256):
  """
Makes predictions given a model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training) 
centersize : a square in the center of the tile is filled with the predicted size
outputimg : the target path in which the predictions are stored
"""
  print("Predicting image "+imagefile)
  data = gdal.Open(imagefile)
  xsize = data.RasterXSize
  ysize = data.RasterYSize
  if(xsize%tilesize == 0):
    xchunks = xsize/tilesize
  else:
    xchunks = (xsize/tilesize)+1
  
  if(ysize%tilesize == 0):
    ychunks = ysize/tilesize
  else:
    ychunks = (ysize/tilesize)+1

  image = standardize_image(data.ReadAsArray())
  #image = zca_whitening(image)
  #image = zca_whitening(data.ReadAsArray())
  #image = standardize_image(image)
  trans       = data.GetGeoTransform()
  proj        = data.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outputimg), data.RasterXSize, data.RasterYSize, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)
  samples = 0
  X = []
  indexes = []
  for i in range(0,xsize,centersize):
    for j in range(0,ysize,centersize): 
      if  i-tilesize/2 >= 0 and i+tilesize/2 < xsize and j-tilesize/2 >= 0 and j+tilesize/2 < ysize:
        X.append(image[:,j-tilesize/2:j+tilesize/2,i-tilesize/2:i+tilesize/2])
        indexes.append((i,j))
        samples+=1
        if samples % batch_size==0:
          X = np.asarray(X)
          preds = model.predict(X,verbose=0)
          for pred,inds in zip(preds,indexes): 
            label, = np.where(pred == max(pred))     
            labelpatch = np.zeros((centersize,centersize))     
            labelpatch.fill(label[0])      
            band = outdata.GetRasterBand(1)
          
            try:
              band.WriteArray(labelpatch, inds[0], inds[1])
            except:
              print( i,j, labelpatch.shape)
          X = []
          indexes = []
          samples = 0
        outdata.FlushCache()
  outdata.FlushCache()



def fit_model(model, generatordata,generatordata_test=None,epocs=20, samples_per_epoch=9000, nb_val_samples=1000,  weightfile="weights.h5",early_stopping=True, patience = 0, save_best=True, monitor='val_acc'):
  """
fit a keras model given a generator for producing image samples
model : the keras model to fit
generatordata : a python generator yielding a pair of numpy arrays compatible with the input and output of the model
epocs : the number of epocs to train 
samples_per_epoch : number of samples to use per epoch
nb_val_samples : number of validation samples to use per epoch
"""
  print("fitting model")
  try:
      callbacks = []
      if early_stopping:
        callbacks.append(EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto'))

      if save_best:
        callbacks.append(ModelCheckpoint("best_"+weightfile, monitor=monitor, verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1))      
        
      if generatordata_test is None:
        history = model.fit_generator(generatordata,
                      steps_per_epoch=samples_per_epoch,
                      validation_steps=nb_val_samples,
                      epochs=epocs,
                      validation_data=generatordata,
                      callbacks=callbacks
                      )
      else:
        history = model.fit_generator(generatordata,
                      steps_per_epoch=samples_per_epoch,
                      validation_steps=nb_val_samples,
                      epochs=epocs,
                      validation_data=generatordata_test,
                      callbacks=callbacks
                      )
      return history
  except KeyboardInterrupt:      
    print("training interrupted by user")
  finally:
    try:
      print("trying to save weights in "+weightfile)
      model.save("final_"+weightfile,overwrite=True)
      print( "weights saved")
    except:
      print( "error saving weights")

def fit_model_on_folder(model, folder, epocs=20, samples_per_epoch=9000, nb_val_samples=10, batch_size=128, weightfile="weights.h5",early_stopping=True, patience = 0, save_best=True, monitor='val_acc', nb_classes=-1):
  """
fit a keras model given a folder containing pairs of image/label image 
model : the keras model to fit
folder : the folder where training images are
epocs : the number of epocs to train 
samples_per_epoch : number of samples to use per epoch
nb_val_samples : number of validation samples to use per epoch
"""
  training_images = []
  training_labels = []
  assert os.path.isdir(folder), "you must specify a valid folder"
  dirList = os.listdir(folder) # current directory
  for dir in dirList:
    if dir.endswith(".tif"):
      if "labels" not in dir:
        assert os.path.isfile(folder+"labels"+dir), folder+"labels"+dir +" does not exist"
        training_labels.append(folder+"labels"+dir) #label images should all have the same name as the raw image and prefix "labels"
        training_images.append(folder+dir)
  assert len(training_labels) == len(training_images), "the folder must contain exactly the same number of images and label images"
  validation_img_ind = np.random.randint(len(training_images))#we chose a validation image randomly among the images in the folder

  print("fitting model")
  try:
      callbacks = []
      if early_stopping:
        callbacks.append(EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto'))

      if save_best:
        callbacks.append(ModelCheckpoint("best_"+weightfile, monitor=monitor, verbose=0,
                 save_best_only=True, save_weights_only=False,
                 mode='auto', period=1))      
      for epoch in range(epocs):
        for i in range(len(training_images)):
          if i != validation_img_ind:
            generatordata, generatordata_test = get_batch_generator(model, training_images[i],training_labels[i], training_images[validation_img_ind], training_labels[validation_img_ind], batch_size,augment_data=True, nb_classes=nb_classes)
            if generatordata_test is None:
              history = model.fit_generator(generatordata,
                            steps_per_epoch=samples_per_epoch,
                            validation_steps=nb_val_samples,
                            epochs=1,
                            validation_data=generatordata,
                            callbacks=callbacks
                            )
            else:
              history = model.fit_generator(generatordata,
                            steps_per_epoch=samples_per_epoch,
                            validation_steps=nb_val_samples,
                            epochs=1,
                            validation_data=generatordata_test,
                            callbacks=callbacks
                            )
      return history
  except KeyboardInterrupt:      
    print("training interrupted by user")
  finally:
    try:
      print("trying to save weights in "+weightfile)
      model.save("final_"+weightfile,overwrite=True)
      print( "weights saved")
    except:
      print( "error saving weights")

def get_batch_generator(model, train_img,train_label_img, validation_img, validation_label_img, batch_size,augment_data=True, test_samples=1000, nb_classes=-1):
  _,img_bands, img_rows, img_cols = model.input_shape
  assert nb_classes != -1, "you should pass the nb_classes argument"
  if len(model.output_shape) == 2: #categorical output
    nb_classes = model.output_shape[1]
    if not os.path.isdir(train_img):#single image
      generatordata = batchGenerator(train_img, 
                                        train_label_img, 
                                        tilesize=img_rows, 
                                        batch_size=batch_size, 
                                        random=True, 
                                        categorical=True,  
                                        nb_classes=nb_classes, 
                                        data_augmentation=augment_data,
                                        )
    else:#folder
      generatordata = batchGeneratorFolder(train_img, 
                                            validation_img, 
                                            tilesize=img_rows, 
                                            batch_size=batch_size, 
                                            random=True, 
                                            categorical=True, 
                                            nb_classes=nb_classes,  
                                            data_augmentation=augment_data)
    generatordata_test = batchGenerator(validation_img, 
                                              validation_label_img, 
                                              tilesize=img_rows, 
                                              batch_size=test_samples, 
                                              random=True, 
                                              categorical=True, 
                                              nb_classes=nb_classes,
                                              data_augmentation=augment_data,
                                              )

  elif len(model.output_shape) == 4 and model.output_shape[1] == 1:
    if not os.path.isdir(train_img):#single image
      generatordata = batchGenerator(train_img, 
                                        train_label_img, 
                                        tilesize=img_rows, 
                                        batch_size=batch_size, 
                                        random=True, 
                                        categorical=False,  
                                        data_augmentation=augment_data,
                                        )
    else:
      generatordata = batchGeneratorFolder(train_img, 
                                        validation_img, 
                                        tilesize=img_rows, 
                                        batch_size=batch_size, 
                                        random=True, 
                                        categorical=False,  
                                        data_augmentation=augment_data,
                                        )
      
    generatordata_test = batchGenerator(validation_img, 
                                              validation_label_img, 
                                              tilesize=img_rows, 
                                              batch_size=test_samples, 
                                              random=True, 
                                              categorical=False, 
                                              data_augmentation=augment_data,
                                              )
  elif len(model.output_shape) == nb_classes+1: #multiple output
    if not os.path.isdir(train_img):#single image
      generatordata = batchGenerator(train_img, 
                                        train_label_img,
                                        tilesize=img_rows,
                                        batch_size=batch_size, 
                                        random=True,
                                        categorical=False,
                                        data_augmentation=False,
                                        nb_classes=nb_classes,
                                        collaborative=True,
                                        )
    else:
      generatordata = batchGeneratorFolder(train_img, 
                                        validation_img,
                                        tilesize=img_rows,
                                        batch_size=batch_size, 
                                        random=True,
                                        categorical=False,
                                        data_augmentation=False,
                                        nb_classes=nb_classes,
                                        collaborative=True,
                                        )
      
    generatordata_test = batchGenerator(validation_img, 
                                              validation_label_img, 
                                              tilesize=img_rows, 
                                              batch_size=1000, 
                                              random=True, 
                                              categorical=False, 
                                              data_augmentation=False, 
                                              nb_classes=nb_classes, 
                                              collaborative=True,
                                              )
  else: #multiple output softmax
    if not os.path.isdir(train_img):#single image
      generatordata = batchGenerator(train_img, 
                                        train_label_img,
                                        tilesize=img_rows,
                                        batch_size=batch_size, 
                                        random=True,
                                        categorical=False,
                                        data_augmentation=False,
                                        nb_classes=nb_classes,
                                        collaborative=True,
                                        softmax=True,
                                        )
    else:
      generatordata = batchGeneratorFolder(train_img, 
                                        validation_img,
                                        tilesize=img_rows,
                                        batch_size=batch_size, 
                                        random=True,
                                        categorical=False,
                                        data_augmentation=False,
                                        nb_classes=nb_classes,
                                        collaborative=True,
                                        softmax=True,
                                        )
    generatordata_test = batchGenerator(validation_img, 
                                              validation_label_img, 
                                              tilesize=img_rows, 
                                              batch_size=1000, 
                                              random=True, 
                                              categorical=False, 
                                              data_augmentation=False, 
                                              nb_classes=nb_classes, 
                                              collaborative=True,
                                              softmax=True,
                                              )
  return generatordata,generatordata_test


def rgb_to_label(rgbimg, outimg, lookup=None):
  """
Converts an rgb tif image into an indexed image in which each color correspond to an integer label
rgbimg : the path to the rgb image to convert
outimg : the target path of the converted image
"""
  img = gdal.Open(rgbimg)
  im = img.ReadAsArray()
  assert len(im.shape) == 3
  c,h,w = im.shape
  assert c == 3
  labelimg = np.zeros((h,w))
  if lookup is None: 
    labelindex = {}
    label = 0
    for i in range(h):
      for j in range(w):
        pix = int(str(im[0][i][j])+str(im[1][i][j])+str(im[2][i][j]))
        if not labelindex.has_key(pix):
          labelindex[pix] = label
          label += 1
        labelimg[i][j] = labelindex[pix]
  else:
    for i in range(h):
      for j in range(w):
        label = (im[0][i][j],im[1][i][j],im[2][i][j])
        if not lookup.has_key(label):
          print("warning unknown label",label)
          labelimg[i][j] = -1
        else:
          labelimg[i][j] = lookup[label]
  
  trans       = img.GetGeoTransform()
  proj        = img.GetProjection()
  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outimg), w, h, 1, gdal.GDT_Float32)
  outdata.SetGeoTransform(trans)
  outdata.SetProjection(proj)  
  band = outdata.GetRasterBand(1)
  band.WriteArray(labelimg, 0, 0)
  outdata.FlushCache()    

def write_array_as_tif(array,outname, trans = None, proj=None,xoff=0,yoff=0, round_values=False):
  l = len(array.shape)
  if l == 3:
    c,h,w = array.shape
  elif l == 2:
    h,w = array.shape
    array = np.asarray([array])
    c = 1
  else:
    print( array.shape, 'incompatible array shape')
    return

  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outname), w, h, c, gdal.GDT_Float32)
  if(trans is not None):
    topx,xres,z,topy,z2,yres = trans
    topx += xoff*xres
    topy += yoff*yres
    trans2 = (topx,xres,z,topy,z2,yres)    
    outdata.SetGeoTransform(trans2)
  if(proj is not None):
    outdata.SetProjection(proj)  
  for ch in range(c):
    band = outdata.GetRasterBand(ch+1)
    if round_values:
      band.WriteArray(np.round(array[ch,:,:]), 0, 0)
    else:
      band.WriteArray(array[ch,:,:], 0, 0)
  outdata.FlushCache() 


def softmax(t):
  shift = t-K.max(t)
  e = K.exp(shift)
  s = tf.expand_dims(K.sum(e,axis=1), 1)+K.epsilon
  return e/s

#y_pred shape should be (S,N,W,H) where S id the number of predicted samples, N is the number of possible classes, W and H are the width and height of the predicted patch
def spatial_cat_crossesntropy(y_true, y_pred):
  epsilon = 1e-6
  s = softmax(y_pred)
  #s = K.clip(s,epsilon,1.0-epsilon)
  return -K.mean(K.sum(y_true*K.log(s), axis=1))
  

def accuracy_one_hot(y_true,y_pred):
  s = softmax(y_pred)
  return K.mean(K.equal(K.argmax(y_true,axis=1), K.argmax(s, axis=1)))
  
def mse_with_unknown(y_true, y_pred):
    '''Computes the MSE ignoring the unknown class (label -1) in the target image'''
    mask = (y_true+1).nonzero()
    error =  y_pred[mask] - y_true[mask]
#    error =  (y_pred - y_true)*(y_true != -1).astype('int')
    return K.mean(K.square(error), axis=-1)



def accuracy_with_unknown(y_true, y_pred):
    '''Computes the MSE ignoring the unknown class (label -1) in the target image'''
    mask = (y_true+1).nonzero()
    return K.mean(K.equal(K.round(y_pred[mask]),K.round(y_true[mask])))

def compute_confusion_matrix(p_img, r_img, offset=0):
  """
Computes the confusion matrix from two labelled images
p_img : the path of the predicted image to evaluate
r_img : the path of the image to use as reference data
offset : the number of pixels to ignore at the border of the images
"""
  predicted = gdal.Open(p_img).ReadAsArray()
  reference = gdal.Open(r_img).ReadAsArray()
  nb_classes = int(max(round(reference.max()),round(predicted.max())))+1
  print( nb_classes)
  matrice = np.zeros((nb_classes,nb_classes))
  
  print( "assuming ", nb_classes, " classes")
  assert reference.min >= 0, "cannot deal with negative labels"
  assert len(predicted.shape)==2, "images should have only 1 channel"  
  assert predicted.shape == reference.shape, "images should have the same dimensions"
  h,w = predicted.shape
  for i in range(offset,h-offset):
    for j in range(offset,w-offset):
      matrice[int(round(predicted[i][j]))][int(round(reference[i][j]))] += 1 
  return matrice      	

def evaluate(p_img, r_img,offset=0, target=-1, channel=0):
  """
Computes accuracy from two labelled images
p_img : the path of the predicted image to evaluate
r_img : the path of the image to use as reference data
"""
  predicted = gdal.Open(p_img).ReadAsArray()
  if len(predicted.shape) == 3: #multi output
    predicted = predicted[channel]
  reference = gdal.Open(r_img).ReadAsArray()
  return np.mean(np.equal(np.round(predicted),np.round(reference)))
           

