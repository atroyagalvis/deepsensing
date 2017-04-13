"""This file provides many miscelaneous functions to load and process GeoTIFF images in order to provide training and testing sets for deep learning with keras"""
import gdal
import numpy as np
np.random.seed(1337)  # for reproducibility
import threading
from keras.preprocessing import image
from keras.callbacks import EarlyStopping


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

"""
generator to produce training image tile couples (image tile, labels) 
imagefile : string containing the path to the raw image to load
vtfile : string containing the path to the target labels, if no parameter passed, the generator will return the same input as target (autoencoder)
tilesize : the size of each generated tile (tiles are always generated as square matrices)
NOTE: imagefile and vtfile should correspond to the exact same area otherwise the training will surely fail
"""
class batchGenerator:
    def __init__(self, imagefile, vtfile='autoencode', tilesize=16, batch_size=32, random=True, categorical=False, nb_classes=-1):
        data = gdal.Open(imagefile)
        self.random = random
        self.categorical = categorical
        self.nb_classes = nb_classes
        self.xsize = data.RasterXSize
        self.ysize = data.RasterYSize
        self.image = standardize_image(data.ReadAsArray())
        if vtfile is not 'autoencode':
          self.autoencode = False 
          self.imagevt = gdal.Open(vtfile).ReadAsArray()
        else:
          self.autoencode = True  
        self.tilesize = tilesize
        self.batch_size = batch_size
        self.lock = threading.Lock()

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
              rx = b+self.tilesize
              ry = b+self.tilesize
            sample = self.image[:,ry:ry+self.tilesize,rx:rx+self.tilesize]
            X.append(sample)
            if self.autoencode:
              y.append(sample)
            elif self.categorical:
              assert self.nb_classes != -1, 'parameter nb_classes should be defined'
              labels = np.zeros(self.nb_classes) 
              #if multiple classes are equally represented we simply pick the first one    
              labels[int(self.imagevt[ry+self.tilesize/2,rx+self.tilesize/2])] = 1
              y.append(labels)
            else:
              y.append(self.imagevt[ry:ry+self.tilesize,rx:rx+self.tilesize])    
          X = np.asarray(X)
          y = np.asarray(y)
          return X,y



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
def predict_image_categorical_center(model,imagefile,tilesize=16, centersize=1,outputimg = "prediction.tif", batch_size=64):
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
def fit_model(model, generatordata,generatordata_test=None,epocs=20, samples_per_epoch=9000, nb_val_samples=1000,  weightfile="weights.h5",early_stopping=True, patience = 0):
  print("fitting model")
  try:
    #for i in xrange(epocs):
     # print("epoch",i)
      if early_stopping:
        callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto')]
      else:
        callbacks = []
      if generatordata_test is None:
        model.fit_generator(generatordata,
                      steps_per_epoch=samples_per_epoch,
                      validation_steps=nb_val_samples,
                      epochs=epocs,
                      validation_data=generatordata,
                      callbacks=callbacks
                      )
      else:
        model.fit_generator(generatordata,
                      steps_per_epoch=samples_per_epoch,
                      validation_steps=nb_val_samples,
                      epochs=epocs,
                      validation_data=generatordata_test,
                      callbacks=callbacks
                      )
  except KeyboardInterrupt:      
    print("training interrupted by user")
  finally:
    try:
      print("trying to save weights in "+weightfile)
      model.save_weights(weightfile,overwrite=True)
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

def write_array_as_tif(array,outname, trans = None, proj=None):
  l = len(array.shape)
  if l == 3:
    c,h,w = array.shape
  if l == 2:
    h,w = array.shape
    array = np.asarray([array])
    c = 1
  outdriver = gdal.GetDriverByName("GTiff")
  outdata   = outdriver.Create(str(outname), w, h, c, gdal.GDT_Float32)
  if(trans is not None):
    outdata.SetGeoTransform(trans)
  if(proj is not None):
    outdata.SetProjection(proj)  
  for ch in xrange(c):
    band = outdata.GetRasterBand(ch+1)
    band.WriteArray(array[ch,:,:], 0, 0)
  outdata.FlushCache() 

  

"""
Computes accuracy from two labelled images
p_img : the path of the predicted image to evaluate
r_img : the path of the image to use as reference data
"""
def evaluate(p_img, r_img):
  predicted = gdal.Open(p_img).ReadAsArray()
  reference = gdal.Open(r_img).ReadAsArray()
  tp = 0.
  assert len(predicted.shape)==2  
  assert predicted.shape == reference.shape
  h,w = predicted.shape
  for i in xrange(h):
    for j in xrange(w):
      if round(predicted[i][j]) == round(reference[i][j]):
        tp +=1.0
  return tp/(w*h)

           

