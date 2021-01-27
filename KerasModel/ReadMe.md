We show the major steps to run the LeNet and other algorithms on Google Colab notebook service.

Step 1: Download this project and move to this folder
```
!git clone https://github.com/worldstar/DeepLearningUsefulCodes.git
%cd DeepLearningUsefulCodes/KerasModel/
```

Step 2: Download dataset and then unzip it quitely.
```
!wget https://github.com/OlafenwaMoses/IdenProf/releases/download/v1.0/idenprof-jpg.zip && unzip -qq idenprof-jpg.zip
```

Step 3: Run LeNet by Keras Funtional API
```
!python lenet_train.py
```
