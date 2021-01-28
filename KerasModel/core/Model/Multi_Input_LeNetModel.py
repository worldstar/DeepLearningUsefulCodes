from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, concatenate, Flatten,Activation,Lambda,Add
from tensorflow.keras import Input, Model
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD


def generator_two_img(dir1,dir2,datagen1,datagen2,img_height,img_width,batch_size):

  train1_generator = datagen1.flow_from_directory(
  dir1,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical')

  train2_generator = datagen2.flow_from_directory(
  dir2,
  target_size=(img_height, img_width),
  batch_size=batch_size,
  class_mode='categorical')

  while True:
    X1i = train1_generator.next()
    X2i = train2_generator.next()
    yield [X1i[0], X2i[0]], X1i[1]


def mish(x):
    return Lambda(lambda x: x*K.tanh(K.softplus(x)))(x)
 
def multi_input_model(img_height, img_width, img_channl,num_classes):

    input1_= Input(shape=(img_height, img_width, img_channl), name='input1')
    input2_ = Input(shape=(img_height, img_width, img_channl), name='input2')

    x1=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x1conv1')(input1_)
    x1=Activation(mish)(x1)
    x1=MaxPooling2D((2,2),strides=(2,2))(x1)
    x1=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x1conv2')(x1)
    x1=MaxPooling2D((2,2),strides=(2,2))(x1)
    x1=Activation(mish)(x1)

    x1=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x1conv3')(x1)
    x1=Activation(mish)(x1)
    x1=MaxPooling2D((2,2),strides=(2,2))(x1)
    x1=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x1conv4')(x1)
    x1=MaxPooling2D((2,2),strides=(2,2))(x1)
    x1=Activation(mish)(x1)
 
    x2 = Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x2conv1')(input2_)
    x2=Activation(mish)(x2)
    x2=MaxPooling2D((2,2),strides=(2,2))(x2)
    x2=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x2conv2')(x2)
    x2=MaxPooling2D((2,2),strides=(2,2))(x2)
    x2=Activation(mish)(x2)
    x2 = Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x2conv3')(x2)
    x2=Activation(mish)(x2)
    x2=MaxPooling2D((2,2),strides=(2,2))(x2)
    x2=Conv2D(32,(5,5),strides=(1,1),padding='valid',name='x2conv4')(x2)
    x2=MaxPooling2D((2,2),strides=(2,2))(x2)
    x2=Activation(mish)(x2)

    x = Add()([x1, x2])
    x = Flatten()(x)

    x=Dense(150,name='fc1')(x)
    x=Activation(mish)(x)
    x=Dense(num_classes,name='fc2')(x)
    x=Dense(num_classes,activation='softmax')(x)
    output_ = Dense(num_classes, activation='sigmoid', name='output')(x)
 
    model = Model(inputs=[input1_, input2_], outputs=[output_])
    model.summary()

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])


 
    return model