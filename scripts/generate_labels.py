import utils

classes = {(255,255,255):0,
           (100,100,100)   :1,
           (0,0,0)      :2,
           (0,0,150):3,
           (0,125,0)    :4,
           (0,255,0)    :5,
           (150,80,0)   :6 
          }
utils.rgb_to_label("./fakeimgeasylabels.tif", "./labelsfakeeasy.tif")
#utils.rgb_to_label("./zh12_GT.tif", "./labelstrain.tif",lookup=classes)
#utils.rgb_to_label("./zh16_GT.tif", "./labelstest.tif",lookup=classes)
