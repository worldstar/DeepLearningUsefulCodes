from core.CustomDataGenerator import CustomDataGenerator
from core.Model.Multi_Input_Model import multi_input_model,generator_two_img

img_height, img_width, img_channl = (256,256,3)

batch_size=32

epochs=2

num_classes = 10

WEIGHTS_FINAL='model-final.h5'

Cusdatagen1=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)
Cusdatagen2=CustomDataGenerator(fun="OTSU",dtype=int)

inputgenerator=generator_two_img(dir1='./idenprof/train',
                dir2='./idenprof/train',
                datagen1=Cusdatagen1,
                datagen2=Cusdatagen2,
                img_height=img_height,
                img_width=img_width,
                batch_size=batch_size)

testgenerator=generator_two_img(dir1='./idenprof/test',
                dir2='./idenprof/test',
                datagen1=Cusdatagen1,
                datagen2=Cusdatagen2,
                img_height=img_height,
                img_width=img_width,
                batch_size=batch_size)

model = multi_input_model(img_height, img_width, img_channl, num_classes)


model.fit(inputgenerator,
    steps_per_epoch=10,
    epochs=epochs,
    validation_data=testgenerator,
    validation_steps=10)

# 儲存訓練好的模型
model.save(WEIGHTS_FINAL)

