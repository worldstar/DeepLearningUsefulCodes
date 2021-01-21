from tensorflow.keras.applications.densenet import DenseNet201,preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
 


def buildLeNetModel(img_height, img_width, img_channl, num_classes):
    inputs = (img_height, img_width, img_channl) 
#base_model = DenseNet(weights='imagenet', include_top=False)
	base_model = DenseNet201(include_top=False)
	 
	x = base_model.output
	x = GlobalAveragePooling2D()(x) 
	predictions = Dense(5, activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	 
	model.summary()
	print('the number of layers in this model:'+str(len(model.layers)))