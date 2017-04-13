"""this script allows to train a model to make remote sensing image analysis given a labelled image wiht the targetted classes and the correspondent unlabelled image, and make predictions given an unlabelled image with the same type as the one used for training a model"""
import numpy as np
np.random.seed(1337)  # for reproducibility
import models
import utils
from keras import optimizers

train_label_img = "./labelszh12.tif"
test_label_img = "./labelszh16.tif"
validation_label_img = './labelszh11c.tif'
train_img = "./zh12.tif"
test_img = "./zh16.tif"
validation_img = './zh11c.tif'
img_bands, img_rows, img_cols, batch_size, nb_classes = 4, 8, 8, 32, 7

#load the VGG16 model fully trainable
model = models.basic_cnn_categorical(img_bands, img_rows, img_cols, nb_classes=nb_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy')
#generate training data (X,y)
generatordata = utils.batchGenerator(train_img, train_label_img, tilesize=img_rows, batch_size=batch_size, random=True, categorical=True, nb_classes=nb_classes)
generatordata_test = utils.batchGenerator(train_img, train_label_img, tilesize=img_rows, batch_size=batch_size, random=True, categorical=True, nb_classes=nb_classes)
#fit the model
utils.fit_model(model,generatordata,generatordata_test,epocs = 1000, samples_per_epoch=30000, nb_val_samples=10,early_stopping=True, patience=5)
weights_path = 'weights.h5'
model.load_weights(weights_path, by_name = True)
#make a prediction on the test image
utils.predict_image_categorical_tile(model,validation_img,tilesize=img_rows, outputimg = "prediction_categorical_center.tif")
print("accuracy : "+str(utils.evaluate("./prediction_categorical_center.tif",validation_label_img)))

