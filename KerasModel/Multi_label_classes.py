from core.Model.LeNet_Sequential_ManyClasses_Model import buildLeNetModel
from core.CustomDataGenerator import CustomDataGenerator
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd
from tensorflow.keras.preprocessing import image 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


epochs = 10
img_height, img_width, img_channl = (256, 256, 3)
num_classes = 6

data = []
labels = []
df = pd.read_csv('./dataset.csv')
# print(df.loc[:,['path']].values)
numdf=int(df.shape[0])

for i in range(numdf):

	image = cv2.imread(df['path'][i])
	image = cv2.resize(image, (img_height, img_width))
	image = img_to_array(image)
	data.append(image)

	l = label=df['labels'][i].split("_")
	labels.append(l)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

(trainX, testX, trainY, testY) = train_test_split(data,
    labels, test_size=0.2, random_state=42)
# print(trainX,'--------',trainY)

datagen=CustomDataGenerator(fun="CLAHE_Color",clahenum=40)
model = buildLeNetModel(img_height, img_width, img_channl, num_classes)

model.fit(datagen.flow(trainX, trainY, batch_size=2),
                    steps_per_epoch=1, epochs=epochs)



