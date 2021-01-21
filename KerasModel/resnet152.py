import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend
import numpy as np
import cv2,os


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
  

  # 資料路徑
  DATASET_PATH  = 'D:/ga/imageAugmentation/data/'

  print(DATASET_PATH)
  # 影像大小
  IMAGE_SIZE = (256, 256)

  # 影像類別數
  NUM_CLASSES = 7

  # 若 GPU 記憶體不足，可調降 batch size 或凍結更多層網路
  BATCH_SIZE = 8

  # 凍結網路層數
  FREEZE_LAYERS = 2

  # Epoch 數
  NUM_EPOCHS =10

  # 模型輸出儲存的檔案
  WEIGHTS_FINAL = 'model-resnet152-final.h5'

  
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

  # 輸出各類別的索引值
  for cls, idx in train_batches.class_indices.items():
      print('Class #{} = {}'.format(idx, cls))

  # 以訓練好的 ResNet152 為基礎來建立模型，
  # 捨棄 ResNet152 頂層的 fully connected layers
  net = tf.keras.applications.ResNet152(include_top=False, weights='imagenet', input_tensor=None,
                 input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],3))
  x = net.output
  x = Flatten()(x)

  # 增加 DropOut layer
  x = Dropout(0.5)(x)

  # 增加 Dense layer，以 softmax 產生個類別的機率值
  output_layer = Dense(NUM_CLASSES, activation='softmax', name='softmax')(x)

  # 設定凍結與要進行訓練的網路層
  net_final = Model(inputs=net.input, outputs=output_layer)
  for layer in net_final.layers[:FREEZE_LAYERS]:
      layer.trainable = False
  for layer in net_final.layers[FREEZE_LAYERS:]:
      layer.trainable = True

  # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
  net_final.compile(optimizer=Adam(lr=1e-5),
                    loss='categorical_crossentropy', metrics=['accuracy'])

  # 輸出整個網路結構
  print(net_final.summary())
  # 訓練模型
  net_final.fit_generator(train_batches,
                          steps_per_epoch = train_batches.samples // BATCH_SIZE,
                          validation_data = valid_batches,
                          validation_steps = valid_batches.samples // BATCH_SIZE,
                          epochs = NUM_EPOCHS)

  # 儲存訓練好的模型
  net_final.save(WEIGHTS_FINAL)