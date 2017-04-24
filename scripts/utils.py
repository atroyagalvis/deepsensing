"""This file provides many miscelaneous functions to load and process GeoTIFF images in order to provide training and testing sets for deep learning with keras"""
import gdal
import numpy as np
np.random.seed(1337)  # for reproducibility
import threading
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


"""
Standardize an image by removing the mean of each band and then dividing by the standard deviation of the band
a : the image to be standardized encoded as a numpy array
"""
def standardize_image(a):
  a = a.astype('float32')
  for i in range(len(a)):
    a[i] -= a[i].mean()
    a[i] /= a[i].std()
  return a

def zca_whitening(x):
  flat_x = np.reshape(x, (x.size,1))
  sigma = np.dot(flat_x.T,flat_x) / flat_x.shape[0]
  u, s, _ = np.linalg.svd(sigma)
  principal_components = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)
  whitex = np.dot(flat_x, principal_components)
  x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
  return x

"""
generator to produce training image tile couples (image tile, labels) 
imagefile : string containing the path to the raw image to load
vtfile : string containing the path to the target labels, if no parameter passed, the generator will return the same input as target (autoencoder)
tilesize : the size of each generated tile (tiles are always generated as square matrices)
NOTE: imagefile and vtfile should correspond to the exact same area otherwise the training will surely fail
"""
class batchGenerator:
    def __init__(self, imagefile, vtfile='autoencode', tilesize=16, batch_size=32, random=True, categorical=False, nb_classes=-1, data_augmentation=True, target_class=-1):
        self.data = gdal.Open(imagefile)
        self.random = random
        self.categorical = categorical
        self.nb_classes = nb_classes
        self.data_augmentation = data_augmentation
        self.xsize = self.data.RasterXSize
        self.ysize = self.data.RasterYSize
        #
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
    def __iter__(self):
        return self

    def next(self):       
        with self.lock:
          X = []
          y = []
          for b in xrange(self.batch_size):
            if self.random:
              rx = np.random.randint(self.xsize-self.tilesize)
              ry = np.random.randint(self.ysize-self.tilesize)    
            else:
              rx = ((self.current_index+b)*self.tilesize)%(self.xsize-self.tilesize)
              ry = ((((self.current_index+b)*self.tilesize)/(self.xsize))*self.tilesize)%(self.ysize-self.tilesize)
              
            sample = self.image[:,ry:ry+self.tilesize,rx:rx+self.tilesize]
            label_img = self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize] 

            if self.data_augmentation:
              lbl = np.asarray([label_img])
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
              labels = label_img
              
                
            
          #  write_array_as_tif(sample, 'test'+str(b)+str(np.random.randint(100))+'.tif', trans = self.data.GetGeoTransform(), proj = self.data.GetProjection(), xoff=rx, yoff=ry )     
            X.append(sample)
            y.append(labels) 
           
          self.current_index += self.batch_size 
          X = np.asarray(X)
          y = np.asarray(y)
          return X,y

"""
perform random transformations on the raw and target images
concatenates the raw image with the target labelles image to apply the 
same random transformation to both images
return the transformed images separatedly
"""
def data_augmentation(raw, labels):
  transform_tensor = np.concatenate((raw,labels))  
  data_augmenter = ImageDataGenerator(rotation_range=180.,
                 width_shift_range=0.05,
                 height_shift_range=0.05,
                 zoom_range=0.5,
                 fill_mode='nearest',
                 horizontal_flip=False,
                 vertical_flip=False,
                 data_format= 'channels_first')
  transform_tensor = data_augmenter.random_transform(transform_tensor)
  raw = transform_tensor[0:len(transform_tensor)-1,:,:]
  labels = transform_tensor[len(transform_tensor)-1,:,:]
  return raw,labels


"""
Makes predictions given a model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training)
outputimg : the target path in which the predictions are stored
"""
def predict_image_tile_new(model,imagefile,tilesize=16,outputimg = "prediction.tif"):
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

  for i in xrange(xchunks):
    for j in xrange(ychunks):
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
      band.WriteArray(pred[0,0:w,0:h], xoff, yoff)
      outdata.FlushCache()
  outdata.FlushCache()

"""
Makes predictions given a model and an raw image file 
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training) each tile is mapped to a single class every pixel of the tile is given the predicted class in the output label map
outputimg : the target path in which the predictions are stored
"""
def predict_image_categorical_tile(model,imagefile,tilesize=16,outputimg = "prediction.tif"):
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

  for i in xrange(0,xsize,tilesize):
    for j in xrange(0,ysize,tilesize): 
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
        print i,j, labelpatch.shape
      outdata.FlushCache()
  outdata.FlushCache()


"""
Makes predictions given a model and an raw image file
model : the keras model used for making the predictions
imagefile : a string corresponding to the path of the raw image to analyse
tilesize : the size of each tile (should correspond to the tile size used for training) 
centersize : a square in the center of the tile is filled with the predicted size
outputimg : the target path in which the predictions are stored
"""
def predict_image_categorical_center(model,imagefile,tilesize=16, centersize=1,outputimg = "prediction.tif", batch_size=256):
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
  for i in xrange(0,xsize,centersize):
    for j in xrange(0,ysize,centersize): 
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
              print i,j, labelpatch.shape
          X = []
          indexes = []
          samples = 0
        outdata.FlushCache()
  outdata.FlushCache()


"""
fit a keras model given a generator for producing image samples
model : the keras model to fit
generatordata : a python generator yielding a pair of numpy arrays compatible with the input and output of the model
epocs : the number of epocs to train 
samples_per_epoch : number of samples to use per epoch
nb_val_samples : number of validation samples to use per epoch
"""
def fit_model(model, generatordata,generatordata_test=None,epocs=20, samples_per_epoch=9000, nb_val_samples=1000,  weightfile="weights.h5",early_stopping=True, patience = 0, save_best=True):
  print("fitting model")
  try:
      callbacks = []
      if early_stopping:
        callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto'))

      if save_best:
        callbacks.append(ModelCheckpoint("best_"+weightfile, monitor='val_loss', verbose=0,
                 save_best_only=True, save_weights_only=True,
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
      model.save_weights("final_"+weightfile,overwrite=True)
      print( "weights saved")
    except:
      print( "error saving weights")

"""
Converts an rgb tif image into an indexed image in which each color correspond to an integer label
rgbimg : the path to the rgb image to convert
outimg : the target path of the converted image
"""
def rgb_to_label(rgbimg, outimg, lookup=None):
  img = gdal.Open(rgbimg)
  im = img.ReadAsArray()
  assert len(im.shape) == 3
  c,h,w = im.shape
  assert c == 3
  labelimg = np.zeros((h,w))
  if lookup is None: 
    labelindex = {}
    label = 0
    for i in xrange(h):
      for j in xrange(w):
        pix = int(str(im[0][i][j])+str(im[1][i][j])+str(im[2][i][j]))
        if not labelindex.has_key(pix):
          labelindex[pix] = label
          label += 1
        labelimg[i][j] = labelindex[pix]
  else:
    for i in xrange(h):
      for j in xrange(w):
        label = (im[0][i][j],im[1][i][j],im[2][i][j])
        if not lookup.has_key(label):
          print "warning unknown label",label
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

def write_array_as_tif(array,outname, trans = None, proj=None,xoff=0,yoff=0):
  l = len(array.shape)
  if l == 3:
    c,h,w = array.shape
  elif l == 2:
    h,w = array.shape
    array = np.asarray([array])
    c = 1
  else:
    print array.shape, 'incompatible array shape'
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
  for ch in xrange(c):
    band = outdata.GetRasterBand(ch+1)
    band.WriteArray(array[ch,:,:], 0, 0)
  outdata.FlushCache() 



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
  print nb_classes
  matrice = np.zeros((nb_classes,nb_classes))
  
  print "assuming ", nb_classes, " classes"
  assert reference.min >= 0, "cannot deal with negative labels"
  assert len(predicted.shape)==2, "images should have only 1 channel"  
  assert predicted.shape == reference.shape, "images should have the same dimensions"
  h,w = predicted.shape
  for i in xrange(offset,h-offset):
    for j in xrange(offset,w-offset):
      matrice[int(round(predicted[i][j]))][int(round(reference[i][j]))] += 1 
  return matrice      	

def evaluate(p_img, r_img,offset=0):
  """
Computes accuracy from two labelled images
p_img : the path of the predicted image to evaluate
r_img : the path of the image to use as reference data
"""
  predicted = gdal.Open(p_img).ReadAsArray()
  reference = gdal.Open(r_img).ReadAsArray()
  tp = 0.
  assert len(predicted.shape)==2  
  assert predicted.shape == reference.shape
  h,w = predicted.shape
  for i in xrange(offset,h-offset):
    for j in xrange(offset,w-offset):
      if round(predicted[i][j]) == round(reference[i][j]):
        tp +=1.0
  return tp/(w*h)

           

