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
predicted_img = "./prediction_residual.tif"
img_bands, img_rows, img_cols, batch_size = 4, 160, 160, 8


#load the VGG16 model fully trainable
model = models.residual_net(img_bands, img_rows, img_cols,nb_blocks=4, res_blocks=[3,4,6,3], filter_depth=[64,128,256,512])

model.compile(optimizer='adam', loss='mse')
#generate training data (X,y)
generatordata = utils.batchGenerator(train_img, train_label_img, tilesize=img_rows, batch_size=batch_size, random=True, categorical=False, data_augmentation=True)
generatordata_test = utils.batchGenerator(train_img, train_label_img, tilesize=img_rows, batch_size=batch_size, random=True, categorical=False,data_augmentation=False)
#fit the model
weights_path = 'weights_resnet.h5'
utils.fit_model(model,generatordata,generatordata_test,epocs = 1000, samples_per_epoch=3000, nb_val_samples=300,early_stopping=True, patience=10, weightfile = weights_path, save_best=True)

model.load_weights(weights_path, by_name = True)
#make a prediction on the test image
utils.predict_image_tile_new(model,validation_img,tilesize=img_rows, outputimg = predicted_img)
print("accuracy : "+str(utils.evaluate(predicted_img,validation_label_img)))

