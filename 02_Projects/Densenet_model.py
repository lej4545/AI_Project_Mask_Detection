from tensorflow import keras # 텐서플로우 임포트
keras.__version__ # 버전 확인

from google.colab import drive # 구글 드라이브에서 구글 colab 연동
drive.mount('/content/drive') # 구글 드라이브 마운트

import os
for dirname, _, filenames in os.walk('/content/drive/My Drive/PART1_ai_project_20210824_0830/02_Projects/03_LeeEunJin'):
    for filename in filenames:
        os.path.join(dirname, filename)

import pandas as pd
import numpy as np
import seaborn as sns
from pandas import DataFrame


## 데이터 들어 있는 폴더
traindir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Train' # 학습할 데이터가 들어있는 폴더 경로
validdir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Validation' #  검증하기 위한 데이터가 들어있는 폴더 경로
testdir='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Test' # 테스트 데이터가 들어있는 폴더 경로

path='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/All'# 모든 이미지가 담겨있는 폴더
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(20,20)) # 잘 불러오는지 확인하기 위해 임의의 5장 사진 출력
for i in range(5):
    file=random.choice(os.listdir(path))
    img_path=os.path.join(path,file)
    image=mpimg.imread(img_path)
    ax=plt.subplot(1,5,i+1)
    plt.imshow(image)

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# train_data, test_data 전처리
train_data=ImageDataGenerator(rescale=1.0/255,
                              zoom_range=0.2, # 20% 확대
                              shear_range=0.2, # 이것은 회전에서 보이지 않는 일종의 '늘이기'를 이미지에 만듭니다
                               rotation_range=40,  # 40도 회전
                              width_shift_range=0.2, # 0.2만큼 옆으로 shift
                              height_shift_range=0.2, # 0.1만큼 위로 shift
                              horizontal_flip=True) # 인풋을 무작위로 가로로 뒤집습니다.


# 이미지를 불러올 때 폴더명에 맞춰 자동으로 labelling 해준다.(2 classes => 'WithMask' : 0 ,'WithoutMask' : 1) 이미지 사이즈는 256 * 256, 배치 사이즈는 32
train_generator = train_data.flow_from_directory(directory=traindir,target_size=(256,256),class_mode='binary', batch_size=32)
test_data=ImageDataGenerator(rescale=1.0/255)
# valid generator 도 train generator와 마찬가지로 진행
valid_generator  test_data.flow_from_directory(directory=validdir,target_size=(256,256),class_mode='binary',batch_size=32)
# test 데이터도 train generator 에서 진행한 방식과 동일하고 추가적으로 rescaling을 진행
test_generator = test_data.flow_from_directory(directory=testdir,target_size=(256,256),class_mode='binary',batch_size=32,shuffle=False)

print(valid_generator.class_indices)


# Densenet 모델 적용

from tensorflow.keras import layers, Sequential
from keras.applications.densenet import DenseNet201 # VDenseNet201모델 적용
from keras.applications.vgg19 import preprocess_input
import tensorflow as tf

# 사전 교육된 기본 모델 생성
densenet201 = DenseNet201(weights='imagenet',include_top=False,input_shape=(256,256,3))

for layer in densenet201.layers:
    layer.trainable = False


vmodel = Sequential() # 모형 작성
vmodel.add(densenet201) # VGG19 기반 모델 추가
vmodel.add(layers.Flatten()) # 평평하게 만들어 Denser 레이어로 전환을 하기 위함.(fully connected되는 부분)
# Dense 첫번째 인자: 출력 뉴런의 수, 두번째 인자 activation 활성화 함수를 설정:
# 'relu' : rectifier 함수로 은닉층에 주로 쓰임. 'sigmoid' : 이진 분류 문제에서 출력 층에 주로 쓰임, 'softmax' : 다중 클래스 분류 문제에서 출력 층에 주로 쓰임.
vmodel.add(layers.Dense(1,activation='sigmoid')) # classcification 하기 위한 부분..

vmodel.summary()

vmodel.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# 데이터 학습.  추후 batch_size 튜닝 필요함
history = vmodel.fit(train_generator,steps_per_epoch=20,epochs=20,validation_data=valid_generator, batch_size = 128)


# 학습된 모델 그래프로 plot하기
def plot_history(history):
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

    plt.title('Loss')
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

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


plot_history(history)

from PIL import Image
import tensorflow as tf

img='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Test/WithMask/2072.jpg'
plt.figure(figsize=(10,10))
image=Image.open(img)
ax=plt.subplot(1,2,1)
plt.imshow(image)
image=np.resize(image,(1,256,256,3))
image = image.astype('float32')
image /= 255


if vmodel.predict(image)[0][0]<0.5:
    print("Mask detected")
else:
    print("No mask detected")


# 학습된 데이터 TEST하기

predictions=vmodel.predict(test_generator)
predictions=np.round(predictions)
y=np.hstack((np.zeros(50),np.ones(50)))

m = tf.keras.metrics.BinaryAccuracy()
m.update_state(y,predictions)
m.result()