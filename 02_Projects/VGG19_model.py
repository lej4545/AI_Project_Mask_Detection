import numpy as np
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_history import plot_history


for dirname, _, filenames in os.walk('/Users/EunJin/PycharmProjects/AI_Project_Mask_Detection/AI_Project_Mask_Detection/01_Images/'):
    for filename in filenames:
        os.path.join(dirname, filename)


## 이미지 데이터 들어 있는 폴더
traindir='/Users/EunJin/PycharmProjects/AI_Project_Mask_Detection/AI_Project_Mask_Detection/01_Images/Train' # 학습할 데이터가 들어있는 폴더 경로
validdir='/Users/EunJin/PycharmProjects/AI_Project_Mask_Detection/AI_Project_Mask_Detection/01_Images/Validation' #  검증하기 위한 데이터가 들어있는 폴더 경로
testdir='/Users/EunJin/PycharmProjects/AI_Project_Mask_Detection/AI_Project_Mask_Detection/01_Images/Test' # 테스트 데이터가 들어있는 폴더 경로

path='/Users/EunJin/PycharmProjects/AI_Project_Mask_Detection/AI_Project_Mask_Detection/01_Images/All'# 모든 이미지가 담겨있는 폴더

plt.figure(figsize=(20,20)) # 잘 불러오는지 확인하기 위해 임의의 5장 사진 출력
for i in range(5):
    file = random.choice(os.listdir(path))
    img_path = os.path.join(path,file)
    image = mpimg.imread(img_path)
    ax = plt.subplot(1,5,i+1)
    plt.imshow(image)

# train_data, test_data 가공
train_data=ImageDataGenerator(rescale=1.0/255,
                              zoom_range=0.2, # 20% 확대
                              shear_range=0.2, # 이것은 회전에서 보이지 않는 일종의 '늘이기'를 이미지에 만듭니다
                              rotation_range=40,  # 40도 회전
                              width_shift_range=0.2, # 0.2만큼 옆으로 shift
                              height_shift_range=0.2, # 0.1만큼 위로 shift
                              horizontal_flip=True) # 인풋을 무작위로 가로로 뒤집습니다.


# 이미지를 불러올 때 폴더명에 맞춰 자동으로 labelling 해준다.(2 classes => 'WithMask' : 0 ,'WithoutMask' : 1) 이미지 사이즈는 256 * 256, 배치 사이즈는 32
train_generator = train_data.flow_from_directory(directory=traindir,target_size=(256,256),class_mode='binary', batch_size=32, shuffle=True)
test_data=ImageDataGenerator(rescale=1.0/255)
# valid generator 도 train generator와 마찬가지로 진행
valid_generator = test_data.flow_from_directory(directory=validdir,target_size=(256,256),class_mode='binary',batch_size=32)
# test 데이터도 train generator 에서 진행한 방식과 동일하고 추가적으로 rescaling을 진행
test_generator = test_data.flow_from_directory(directory=testdir,target_size=(256,256),class_mode='binary',batch_size=32,shuffle=False)

print(valid_generator.class_indices)

# VGG19 모델 적용
from tensorflow.keras import layers, Sequential
from keras.applications.vgg19 import VGG19 # VGG19 모델 적용

# pre-trained base model 가져오기
vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(256,256,3))

# Freeze the base model
for layer in vgg19.layers:
    layer.trainable = False

vmodel = Sequential() # 모형 작성
vmodel.add(vgg19) # VGG19 기반 모델 추가
vmodel.add(layers.Flatten())
 # 평평하게 만들어 Denser 레이어로 전환을 하기 위함.(fully connected되는 부분)
# Dense 첫번째 인자: 출력 뉴런의 수, 두번째 인자 activation 활성화 함수를 설정:
# 'relu' : rectifier 함수로 은닉층에 주로 쓰임. 'sigmoid' : 이진 분류 문제에서 출력 층에 주로 쓰임, 'softmax' : 다중 클래스 분류 문제에서 출력 층에 주로 쓰임.
vmodel.add(layers.Dense(1,activation='sigmoid')) # classcification 하기 위한 부분..

vmodel.summary()

vmodel.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# 데이터 학습.
history = vmodel.fit(train_generator,steps_per_epoch=len(train_generator),epochs=20,validation_data=valid_generator)

vmodel.save('saved_model_vgg19.h5')

# 학습된 모델 plot
# plot_history(history)

# img='/content/drive/My Drive/PART1_ai_project_20210824_0830/01_Images/Test/WithMask/2072.jpg'
# plt.figure(figsize=(10,10))
# image=Image.open(img)
# ax=plt.subplot(1,2,1)
# plt.imshow(image)
# image=np.resize(image,(1,256,256,3))
# image = image.astype('float32')
# image /= 255
#
#
# if vmodel.predict(image)[0][0]<0.5:
#     print("Mask detected")
# else:
#     print("No mask detected")
#
# predictions=vmodel.predict(test_generator)
# predictions=np.round(predictions)
# y=np.hstack((np.zeros(483),np.ones(992-483)))
#
# m = tf.keras.metrics.BinaryAccuracy()
# m.update_state(y,predictions)
# m.result()