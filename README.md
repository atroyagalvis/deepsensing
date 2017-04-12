# deepsensing
A set of scripts to perform remote sensing image interpretation using deep learning.

Remote sensing image interpretation consists in analysing an image taken by an aerial or sattelite engine in order to associate to each pixel a sematic class related to a given application domain.

Deepsensing is a set of scripts to:
  -read GEOTIFF images using gdal
  -write a numpy array as a GEOTIFF image
  -build, and train deep learning models using the keras framework
  -evaluate the quality of a predicted image with respect to a reference image
  -more
  
To train a model you need a ready-to-analyse remote sensing image, an label image corresponding to the same area containing the targetted classes.


Requirements:
  Python2.7 with keras, and gdal modules installed


example: (to run on GPU using theano)
THEANO_FLAGS=floatX=float32,device=cuda0,nvcc.flags=-D_FORCE_INLINES python deepsensing_categorical.py 
