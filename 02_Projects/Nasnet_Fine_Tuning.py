from tensorflow import keras # 텐서플로우 임포트
keras.__version__ # 버전 확인

from google.colab import drive # 구글 드라이브에서 구글 colab 연동

drive.mount('/content/drive') # 구글 드라이브 마운트

import os # os모듈 임포트
for dirname, _, filenames in os.walk('/content/drive/My Drive/PART1_ai_project_20210824_0830/02_Projects/02_MinSeungJun'):
    for filename in filenames: # 아마도 위에 폴더 안에 있는 파일들을 불러와서
        os.path.join(dirname, filename) # 폴더 이름과 파일 이름을 합쳐주는 것 같다.

import pandas as pd # 필요한 묘둘 임포트
import numpy as np
import seaborn as sns
from pandas import DataFrame

traindir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Train' # 학습할 데이터가 들어있는 폴더 경로
validdir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Validation' #  검증하기 위한 데이터가 들어있는 폴더 경로
testdir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Test' # 테스트 데이터가 들어있는 폴더 경로

path='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/All'# 모든 이미지가 담겨있는 폴더
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img #imagedatagenerator 임포트

# train_data, test_data 전처리
train_data=ImageDataGenerator(rescale=1.0/255,
                              zoom_range=0.2, # 20% 확대
                              shear_range=0.2, # 이것은 회전에서 보이지 않는 일종의 '늘이기'를 이미지에 만듭니다
                               rotation_range=40,  # 40도 회전
                              width_shift_range=0.2, # 0.2만큼 옆으로 shift
                              height_shift_range=0.2, # 0.1만큼 위로 shift
                              horizontal_flip=True)


# 이미지를 불러올 때 폴더명에 맞춰 자동으로 labelling 해준다.('WithMask' : 0 ,'WithoutMask' : 1) 이미지 사이즈는 256* 256, 배치 사이즈는 32
train_generator = train_data.flow_from_directory(directory=traindir,target_size=(331, 331),class_mode='binary',batch_size=32)
test_data=ImageDataGenerator(rescale=1.0/255)
# valid generator 도 train generator와 마찬가지로 진행
valid_generator = test_data.flow_from_directory(directory=validdir,target_size=(331, 331),class_mode='binary',batch_size=32)
# test 데이터도 train generator 에서 진행한 방식과 동일하고 추가적으로 rescaling을 진행
test_generator = test_data.flow_from_directory(directory=testdir,target_size=(331, 331),class_mode='binary',batch_size=32,shuffle=False)

print(valid_generator.class_indices)

from tensorflow.keras import layers, Sequential
from keras.applications.nasnet import NASNetLarge  # VGG19 모델 적용
from keras.applications.nasnet import preprocess_input

NasNet = NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))

for layer in NasNet.layers:
    layer.trainable = False

vmodel_0 = Sequential()
vmodel_0.add(NasNet)
vmodel_0.add(layers.Flatten())
vmodel_0.add(layers.Dense(1, activation='sigmoid'))
vmodel_0.summary()

vmodel_0.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# 데이터 학습.
history_0 = vmodel_0.fit(train_generator,steps_per_epoch=len(train_generator),epochs=15,validation_data=valid_generator, batch_size = 128)


# 학습된 모델 그래프로 plot하기
def plot_history(history, LR):
    LR = str(LR)
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss (LR = ' + LR + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy (LR = ' + LR + ')')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_history(history_0,0.0001)

# LR = 0.0001 에 대한 TEST 값
predictons_0 = history_0.predict(test_generator)
predictions_0 = np.round(predictons_0)

m = tf.keras.metrics.BinaryAccuracy()
y = np.hstack((np.zeros(50),np.ones(50)))
m.update_state(y, predictions_0)
m.result()