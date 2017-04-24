"""this script allows to train a model to make remote sensing image analysis given a labelled image wiht the targetted classes and the correspondent unlabelled image, and make predictions given an unlabelled image with the same type as the one used for training a model"""
import numpy as np
np.random.seed(1337)  # for reproducibility
import models
import utils
from keras import optimizers
import matplotlib.pyplot as plt

test_label_img = "./labelszh12.tif"
train_label_img = "./labelszh16.tif"
validation_label_img = './labelszh11c.tif'
test_img = "./zh12.tif"
train_img = "./zh16.tif"
validation_img = './zh11c.tif'
predicted_img = "./prediction_residual.tif"
rows, cols = 1146, 872 
img_bands, img_rows, img_cols, batch_size, nb_classes = 4, 32, 32, 512, 7


#load the model
#model = models.residual_net(img_bands, img_rows, img_cols,nb_blocks=3, res_blocks=[3,2,2], filter_depth=[64,128,256], categorical=True, nb_classes= nb_classes)
model = models.conv_pool_dense_net(img_bands, img_rows, img_cols,nb_blocks=3, dense_layers=3, in_blocks=[3,2,1], filter_depth=[64,128,256], categorical=True, nb_classes= nb_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy')

#generate training data (X,y)
generatordata = utils.batchGenerator(train_img, train_label_img, tilesize=img_rows, batch_size=batch_size, random=True, categorical=True,  nb_classes=nb_classes, data_augmentation=True)
generatordata_test = utils.batchGenerator(validation_img, validation_label_img, tilesize=img_rows, batch_size=500, random=True, categorical=True, nb_classes=nb_classes,data_augmentation=False)
data_test = generatordata_test.next()
#fit the model
weights_path = 'weights_resnet.h5'
#"""
history = utils.fit_model(model,generatordata,data_test,epocs = 1000, samples_per_epoch=(rows*cols)/batch_size, nb_val_samples=10,early_stopping=True, patience=1, weightfile = weights_path, save_best=True)

print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#"""
model.load_weights('best_'+weights_path, by_name = True)
#make a prediction on the test image
utils.predict_image_categorical_center(model,test_img,tilesize=img_rows, outputimg = predicted_img, centersize=1)
print("accuracy : "+str(utils.evaluate(predicted_img,test_label_img, offset= img_rows/2)))

#

