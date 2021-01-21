import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import cv2,os
from core.Model.LeNet_Sequential import buildLeNetModel
from tensorflow.keras.callbacks import ModelCheckpoint

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)
        # # logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        # # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # devices = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(devices[0], True)

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class CustomDataGenerator(ImageDataGenerator):

    def __init__(self,
                fun="",
                clahenum=None,
                h=None,
                kernel=None,
                **kwargs):
        '''
        Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        '''
        self.fun = fun #fun參數
        self.clahe_num = clahenum #clahe參數
        self.h = h #NLM參數
        self.kernel = kernel #open參數

        if self.fun == "NLM":
            function=self.NLM
        if self.fun == "CLAHE_Color":
            function=self.CLAHE_Color
        if self.fun == "Opening_operation":
            function=self.Opening_operation
        if self.fun == "OTSU":
            function=self.OTSU

        
        super().__init__(
            preprocessing_function=function,
            **kwargs)
       

    def NLM(self, image):
        '''
        h:決定濾波器強度。較高的值可以更好的消除噪聲，但也會刪除圖像細節(10的效果比較好)
        hForColorComponents:與h相同，但只適用於彩色圖像(該值通常與h相同)
        templateWindowSize:奇數(推薦值為7)
        searchWindowSize:奇數(推薦值為21)
        '''
        # self.i=self.i+1

        temp=random.randint(0, 40)
        img = image.astype(np.uint8) # convert to int
        dst = cv2.fastNlMeansDenoisingColored(img,None,self.h,self.h,7,21)

        # f=('./test/NLM{}.png'.format(self.i))
        # cv2.imwrite(f,dst)

        return dst

    def CLAHE_Color(self,image):
        # self.i=self.i+1

        img = image.astype(np.uint8) # convert to int
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # f=('./test/CLAHE_Color{}.png'.format(self.i))
        # cv2.imwrite(f,final)

        return final

    def Opening_operation(self,image):

        # self.i=self.i+1
        
        kernel = np.ones((self.kernel,self.kernel),np.uint8) 
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        # f=('./test/Opening_operation{}.png'.format(self.i))
        # cv2.imwrite(f,opening)

        return opening
    
    def OTSU(self,image):
        # self.i=self.i+1

        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # ret2,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # f=('./test/OTSU{}.png'.format(self.i))
        # cv2.imwrite(f,binary)

        return binary




if __name__ == "__main__":

    log_dir="./model/"
    img_height, img_width, img_channl = (256,256,3) #224, 224, 3
    batch_size=32
    epochs=10
    num_classes = 7
    # datagen=ImageDataGenerator(preprocessing_function = CLAHE_Color,dtype=int)
    '''
        [fun]           [param]
     CLAHE_Color        clahenum
     NLM                h
     Opening_operation  kernel
     OTSU
    '''
    datagen=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)

    train_generator = datagen.flow_from_directory(
        './DCdata/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    val_generator = datagen.flow_from_directory(
            './DCdata/val',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model = buildLeNetModel(img_height, img_width, img_channl, num_classes)

    

    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    callbacks_list = [checkpoint]

    model.fit(
        train_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10, callbacks=[callbacks_list])
