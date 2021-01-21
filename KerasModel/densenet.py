import random
import cv2,os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

class CustomDataGenerator(ImageDataGenerator):

    def __init__(self,
                fun="",
                clahenum=None,
                h=None,
                kernel=None,
                **kwargs):

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


        temp=random.randint(0, 40)
        img = image.astype(np.uint8) # convert to int
        dst = cv2.fastNlMeansDenoisingColored(img,None,self.h,self.h,7,21)



        return dst

    def CLAHE_Color(self,image):


        img = image.astype(np.uint8) # convert to int
        lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)  
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(self.clahe_num,self.clahe_num))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


        return final

    def Opening_operation(self,image):

        
        kernel = np.ones((self.kernel,self.kernel),np.uint8) 
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


        return opening
    
    def OTSU(self,image):

        ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

        return binary



if __name__ == "__main__":

  # 資料路徑
  DATASET_PATH  = 'D:/ga/imageAugmentation/data/'

  IMAGE_SIZE = (256, 256)

  BATCH_SIZE=32

  NUM_EPOCHS=10

  NUM_CLASSES = 7

  WEIGHTS_FINAL = 'model-densenet-final.h5'


  datagen=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)

  train_batches = datagen.flow_from_directory(DATASET_PATH + '/Train',
                                                    target_size=IMAGE_SIZE,
                                                    interpolation='bicubic',
                                                    class_mode='categorical',
                                                    shuffle=True,
                                                    batch_size=BATCH_SIZE)


  valid_batches = datagen.flow_from_directory(DATASET_PATH + '/val',
                                                    target_size=IMAGE_SIZE,
                                                    interpolation='bicubic',
                                                    class_mode='categorical',
                                                    shuffle=False,
                                                    batch_size=BATCH_SIZE)



  # model = tf.keras.applications.DenseNet121(
  # include_top=True,
  # # weights="imagenet",
  # input_tensor=None,
  # input_shape=None,
  # pooling=None,
  # # classes=1000,
  # )

  model = DenseNet201(include_top=False)

  # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
  model.compile(optimizer=Adam(lr=1e-5),
                    loss='categorical_crossentropy', metrics=['accuracy'])

  # 輸出整個網路結構
  print(model.summary())
  # 訓練模型
  model.fit_generator(train_batches,
                          steps_per_epoch = train_batches.samples // BATCH_SIZE,
                          validation_data = valid_batches,
                          validation_steps = valid_batches.samples // BATCH_SIZE,
                          epochs = NUM_EPOCHS)

  # 儲存訓練好的模型
  model.save(WEIGHTS_FINAL)