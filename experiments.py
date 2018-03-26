import numpy as np
np.random.seed(1337)  # for reproducibility
import models
import utils
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt
import gdal
import csv
import pickle

train_on_folder = False
train = True

training_folder = "./data/train/"
testing_folder = "./data/test/"
test_label_img = "data/test/labelszh11.tif"#"img_Strasbg/labels_strasbourg_test.tif"#
train_label_img = "data/train/labelszh16.tif"#"img_Strasbg/labels_strasbourg_reliable.tif"#"img_Strasbg/labels_strasbourg_train.tif"#
validation_label_img = 'data/train/labelszh12.tif'#"img_Strasbg/labels_strasbourg_validation.tif" #
test_img = "data/test/zh11.tif"#"img_Strasbg/strasbourg_test.tif"#"img_Strasbg/strasbourg_test.tif"#
train_img = "data/train/zh16.tif"#"img_Strasbg/strasbourg.tif"#"img_Strasbg/strasbourg_train.tif"#
validation_img = 'data/train/zh12.tif'#"img_Strasbg/strasbourg_validation.tif"#

predicted_img = "prediction_residual.tif"
#trainlabels = gdal.Open(train_label_img).ReadAsArray()
#rows, cols = trainlabels.shape
img_bands, img_rows, img_cols, batch_size, nb_classes = 4, 32, 32, 128, 9
convs, blocs, filters = 3, [4,3,2], [64,128,256]
droprate = 0.5
augment_data = True
test_samples = 2000
epochs = 1000
train_samples = 1000
expe_prefix = "fulldata"
#compute class proportions to weight losses
#hist,_ = np.histogram(trainlabels, bins=range(nb_classes+1))
#weights = list(1-(hist/float(hist.sum())))
#weights.append(1.)

def set_regularization(model):
  for l in model.layers:
    l.kernel_regularizer=regularizers.l2(0.01)
    l.activity_regularizer=regularizers.l1(0.01)

print( 'instantiating models')
#instantiate all tested models
tested_models = {}
#tested_models['full_dense'] = models.full_dense(img_bands, img_rows, img_cols, hidden_layers=convs, categorical=False, droprate=droprate)
#tested_models['full_dense_cat'] = models.full_dense(img_bands, img_rows, img_cols, categorical=True,  hidden_layers=convs, nb_classes=nb_classes, droprate=droprate)
#tested_models['conv_pool_dense'] = models.conv_pool_dense_net(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)
#tested_models['conv_pool_dense_cat'] = models.conv_pool_dense_net(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1,categorical=True,nb_classes=nb_classes, droprate=droprate)
#tested_models['residual'] = models.residual_net(img_bands, img_rows, img_cols,nb_blocks=convs, res_blocks=blocs, filter_depth=filters, categorical=False, nb_classes=nb_classes, droprate=droprate)
#tested_models['residual_cat'] = models.residual_net(img_bands, img_rows, img_cols,nb_blocks=convs, res_blocks=blocs, filter_depth=filters, categorical=True, nb_classes=nb_classes, droprate=droprate)
#tested_models['conv_bn'] = models.conv_pool_dense_net_BN(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)
#tested_models['conv_bn_cat'] = models.conv_pool_dense_net_BN(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1,categorical=True,nb_classes=nb_classes, droprate=droprate)
#tested_models['conv_transconv'] = models.conv_pool_transconv_net(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)
#tested_models['conv_transconv_cat'] = models.conv_pool_transconv_net(img_bands, img_rows, img_cols, nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1,categorical=True,nb_classes=nb_classes, droprate=droprate)
#tested_models['conv_multi'] = models.collaborative_net(img_bands, img_rows, img_cols, nb_classes=nb_classes,nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)
#tested_models['conv_multi_softargmax'] = models.collaborative_net_softargmax(img_bands, img_rows, img_cols, nb_classes=nb_classes,nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)
tested_models['conv_multi_softmax'] = models.collaborative_net_softmax(img_bands, img_rows, img_cols, nb_classes=nb_classes,nb_blocks=convs, in_blocks=blocs, filter_depth=filters, dense_layers=1, categorical=False,droprate=droprate)

for modelname in tested_models:
  model = tested_models[modelname]
  set_regularization(model)



if train:
  training_histories = {}
  for modelname in tested_models:
    model = tested_models[modelname]
    opt = optimizers.Adam(lr=0.0001)
    if len(model.output_shape) == 2: #categorical output
      model.compile(optimizer=opt, loss='categorical_crossentropy',metrics = [utils.accuracy_with_unknown])
      monitor_metric='val_accuracy_with_unknown'
      
    elif len(model.output_shape) == 4 and model.output_shape[1] == 1:
      model.compile(optimizer=opt, loss=utils.mse_with_unknown,metrics = [utils.accuracy_with_unknown])
      monitor_metric='val_accuracy_with_unknown'
      
    elif len(model.output_shape) == nb_classes+1: #multiple output
      model.compile(optimizer=opt, loss=utils.mse_with_unknown,metrics = [utils.accuracy_with_unknown])
      monitor_metric='val_out_accuracy_with_unknown'
      
    else: #multiple output softmax
      model.compile(optimizer=opt, loss=utils.spatial_cat_crossesntropy,metrics = [utils.accuracy_one_hot])
      monitor_metric='val_accuracy_one_hot'
    
    print( 'train :', modelname    )
    weights_path = expe_prefix+modelname+".h5" 
    if train_on_folder:
      print( "train on folder")
      generatordata, generatordata_test = utils.get_batch_generator(model, training_folder,None, validation_img, validation_label_img, batch_size,augment_data=True, nb_classes=nb_classes)
    else:
      print ("train with one image")
      generatordata, generatordata_test = utils.get_batch_generator(model, train_img,train_label_img, validation_img, validation_label_img, batch_size,augment_data=True, nb_classes=nb_classes)
    data_test = next(generatordata_test)
    history = utils.fit_model(model,generatordata,data_test,epocs = epochs, samples_per_epoch=train_samples, nb_val_samples=test_samples,early_stopping=False, weightfile = weights_path, save_best=True,monitor=monitor_metric)  
    
    training_histories[modelname] = history

      
  print (training_histories)
  historyfile = expe_prefix+'histories.csv'
  with open(historyfile, 'wb') as csv_file:
      writer = csv.writer(csv_file)
      for m, h in  training_histories.items():       
        for key, value in h.history.items():
          writer.writerow([m+key, value])

#utils.fit_model_on_folder(None, "./img_zurich/", epocs=20, samples_per_epoch=9000, nb_val_samples=1000, batch_size=128, weightfile="weights.h5",early_stopping=True, patience = 0, save_best=True, monitor='val_acc')
historyfile = expe_prefix+'histories.csv'
#visualize_history.visualize_history(models = tested_models.keys(), historyfile = historyfile)
predict.predict_folder(testing_folder,models = tested_models.keys(),result_prefix = 'best_', expe_prefix = expe_prefix, folder="./", nb_classes=nb_classes)

#predict.predict_with_models(validation_label_img = test_label_img, validation_img = test_img, models = tested_models.keys(),result_prefix = 'best_', expe_prefix = expe_prefix, folder="./", nb_classes=nb_classes)
#predict.predict_with_models(validation_label_img = validation_label_img, validation_img = validation_img, models = tested_models.keys(),result_prefix = 'final_', expe_prefix = expe_prefix)

#To read it back:

#with open(historyfile, 'rb') as csv_file:
 #   reader = csv.reader(csv_file)
  #  mydict = dict(reader)

