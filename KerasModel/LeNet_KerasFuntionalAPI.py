import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
import cv2,os
from core.Model.LeNet_Functional import buildLeNetModel


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

        self.i=0
        
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
        temp=random.randint(0, 40)
        img = image.astype(np.uint8) # convert to int
        dst = cv2.fastNlMeansDenoisingColored(img,None,self.h,self.h,7,21)

        return dst

    def CLAHE_Color(self,image):
        self.i=self.i+1
        img = image.astype(np.uint8) # convert to int
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        f=('./test/NLM{}.png'.format(self.i))
        cv2.imwrite(f,final)

        return final

    def Opening_operation(self,image):

        kernel = np.ones((self.kernel,self.kernel),np.uint8) 
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        return opening
    
    def OTSU(self,image):

        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        return binary

if __name__ == "__main__":
    
    log_dir="./model/"
    inputs=(150,150,3)
    batch_size=32
    epochs=10
    num_classes = 2

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

    model = buildLeNetModel(inputs, num_classes)

    checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    callbacks_list = [checkpoint]

    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10, callbacks=[callbacks_list])

