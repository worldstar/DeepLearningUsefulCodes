from sklearn.metrics import confusion_matrix,accuracy_score,precision_score, recall_score, f1_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from core.CustomDataGenerator import CustomDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import tensorflow
import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from core.Model.LeNet_Functional_Model import buildLeNetModel,mish
from sklearn.preprocessing import LabelEncoder

def readimg(imagePaths):

    data=[]
    labels=[]

    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list

        image = cv2.imread(imagePath)
        image = cv2.resize(image, (img_height, img_width))
        image2 = img_to_array(image)
        data.append(image2)

        l = label = imagePath.split(os.path.sep)[-2].split("\\")
        labels.append(l)

    data = np.array(data, dtype="float") / 255.0
    labels = np.asarray(labels)

    encoder = LabelEncoder()
    encoder.fit(labels)
    labels = to_categorical(encoder.transform(labels))


    return data,labels

if __name__ == "__main__":

    dataphotoval='./data/val/'
    loadModelPath="./model/1.LeNet-model-final.h5"
    img_height, img_width, img_channl = (256, 256, 3)

    imagePathsval = sorted(list(paths.list_images(dataphotoval)))

    model = load_model(loadModelPath,custom_objects={'mish':mish, 'buildLeNetModel':buildLeNetModel})

    os.environ["PATH"] += os.pathsep +'C:/Program Files/Graphviz/bin' 
    tensorflow.keras.utils.plot_model(model, show_shapes=True)
    # model = load_model(WEIGHTS_FINAL)
    testX,testY=readimg(imagePathsval)

    predictions = model.predict(testX)

    y_pred_unencoded = np.argmax(predictions, axis=1)
    y_test_unencoded = np.argmax(testY, axis=1)

    accuracyscore=accuracy_score(y_test_unencoded, y_pred_unencoded)
    print('accuracyscore:',accuracyscore)

    print('confusion_matrix-')
    print(confusion_matrix(y_test_unencoded, y_pred_unencoded))

    print('Precision:', precision_score(y_test_unencoded, y_pred_unencoded, average='macro'))
    print('Recall:',recall_score(y_test_unencoded, y_pred_unencoded, average='macro'))
    print('F1:',f1_score(y_test_unencoded, y_pred_unencoded, average='macro'))



'''
如果要使用plot_model須先安裝pydot和graphviz這兩個函數庫 並到(https://www.graphviz.org/download/)下載Windows版本的graphviz軟體

{注意!請依照自己的電腦環境選擇64-bit或32-bit} 
點選->2.46.1 EXE installer for Windows 10 (64-bit): stable_windows_10_cmake_Release_x64_graphviz-install-2.46.1-win64.exe (not all tools and libraries are included)

下載完成後一直按下一步就可以了,之後將路經加入系統變量(https://www.cnblogs.com/travelcat/p/11429437.html)可以透過此網址查看如何新增系統變量
之後使用可以在當前路徑model.png查看。
'''