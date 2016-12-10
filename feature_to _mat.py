
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import scipy.io as sio

dist={}
base_model = VGG19(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)
for i in range(99):
	
	img_path = 'sel_frame/'+str(i)+'.jpg'
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	fc2_feature = model.predict(x)
	dist[str(i)]=fc2_feature[0]
#print dist
sio.savemat('feature.mat',dist)

