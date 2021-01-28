from core.CustomDataGenerator import CustomDataGenerator
from core.Model.Multi_Input_Model import multi_input_model
import pandas as pd

if __name__ == "__main__":
	
    inputs=(150,150,3)
    batch_size=32
    epochs=10
    num_classes = 2

    datagen1=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)
    datagen2=CustomDataGenerator(fun="OTSU",dtype=int)
    WEIGHTS_FINAL='model-final.h5'
    train_dir='./DCdata/train'

    def generator_two_img():
        train1_generator = datagen1.flow_from_directory(
        './DCdata/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')

        train2_generator = datagen2.flow_from_directory(
        './DCdata/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
        return train1_generator

        # genX1 = gen.flow(X1, y,  batch_size=batch_size, seed=1)
        # genX2 = gen.flow(X2, y, batch_size=batch_size, seed=1)
    	while True:
    	    X1i = train1_generator.next()
    	    X2i = train2_generator.next()
    	    yield [X1i[0], X2i[0]], X1i[1]

    train1_generator = datagen1.flow_from_directory(
        './DCdata/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')


    val_generator = datagen2.flow_from_directory(
            './DCdata/val',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    model = multi_input_model(inputs, num_classes)


    model.fit(train1_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10)

    # 儲存訓練好的模型
    model.save(WEIGHTS_FINAL)

