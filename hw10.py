from tensorflow.keras import Model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import random

def load_data():
    # 먼저 MNIST 데이터셋을 로드하겠습니다.
    # 케라스는 `keras.datasets`에 널리 사용하는 데이터셋을 로드하기 위한 함수를 제공합니다.
    # 이 데이터셋은 이미 훈련 세트와 테스트 세트로 나누어져 있습니다.
    # 훈련 세트를 더 나누어 검증 세트를 만드는 것이 좋습니다:

    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data() # data를 가져옵니다.
    X_train_full = X_train_full.astype(np.float32) # X_train_full을 float 형태로 변환합니다.
    X_test = X_test.astype(np.float32) # X_train을 float 형태로 변환합니다.
    #print(X_train_full.shape, y_train_full.shape)
    #print(X_test.shape, y_test.shape)
    return X_train_full, y_train_full, X_test, y_test # Data를 반환합니다.

def data_normalization(X_train_full, X_test):
    # 전체 훈련 세트를 검증 세트와 (조금 더 작은) 훈련 세트로 나누어 보죠. 또한 픽셀 강도를 255로 나누어 0~1 범위의 실수로 바꾸겠습니다.

    X_train_full = X_train_full / 255. #정규화 하기위해 255로 나누어줌

    X_test = X_test / 255. #정규화 하기위해 255로 나누어줌
    train_feature = np.expand_dims(X_train_full, axis=3) #Data format을 NHWC형태로 만들어줌
    test_feature = np.expand_dims(X_test, axis=3) #Data format을 NHWC형태로 만들어줌

    print(train_feature.shape, train_feature.shape) #바뀐 형태 shape확인
    print(test_feature.shape, test_feature.shape) #바뀐 형태 shape확인

    return train_feature,  test_feature #Data를 리턴


def draw_digit(num): #그림으로 표현하기
    for i in num: 
        for j in i:
            if j == 0:
                print('0', end='')
            else :
                print('1', end='')
        print()

def makemodel(X_train, y_train, X_valid, y_valid, weight_init):
    model = Sequential() # Sequential 모델생성
    model.add(Conv2D(32, kernel_size=(3, 3),  input_shape=(28,28,1), activation='relu')) # kernel_size 3x3, 들어가는 Data는 (28,28,1), relu 사용하여 padding하여 32필터생성
    model.add(MaxPooling2D(pool_size=2)) #2x2size로 MaxPooling 하여 크기를 줄임
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # kernel_size 3x3, 들어가는 Data는 (28,28,1), relu 사용하여 padding 하여 64필터생성
    model.add(Dropout(0.25)) #Dropout 25%로 과적합을 막음
    model.add(Flatten()) #만들어진 데이터를 직렬화함
    model.add(Dense(128, activation='relu')) #128개의 relu 히든레이어
    model.add(Dense(10, activation='softmax')) # 0~9로 softmax
    model.summary() #summary
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])#다음 loss,optimizer, metrics등에 따라 모델컴파일



    return model # model 리턴

def plot_history(histories, key='accuracy'): #그림으로 표현
    plt.figure(figsize=(16,10)) #사이즈는 16,10
    #그림제작 부분
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')
    
    plt.xlabel('Epochs') #xlable은 Epochs
    plt.ylabel(key.replace('_',' ').title()) #ylabel은 key.replace('_',' ')의 타이틀사용
    plt.legend() #라벨 표현

    plt.xlim([0,max(history.epoch)]) #x의 범위는 0 ~ epoch최대치로 지정
    plt.show() #그림 표현

def draw_prediction(pred, k,X_test,y_test,yhat): #예측
    samples = random.choices(population=pred, k=16)#랜덤 선택

    count = 0 #변수설정
    nrows = ncols = 4 #변수설정
    plt.figure(figsize=(12,8))#(12,8)로 그림설정

    for n in samples:
        count += 1 #count 증가
        plt.subplot(nrows, ncols, count) # (4,4)에 그림표현
        plt.imshow(X_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')#이미지 표현
        tmp = "Label:" + str(y_test[n]) + ", Prediction:" + str(yhat[n])#라벨작성
        plt.title(tmp)#라벨등록

    plt.tight_layout() 
    plt.show()#그림 표현

def evalmodel(X_test,y_test,model):
    yhat = model.predict(X_test) #x_test예측
    yhat = yhat.argmax(axis=1) #max값을 yhat에 넣음

    print(yhat.shape) #shape확인
    answer_list = [] #리스트 선언
    # yhat[n] == y_test[n]일시 즉 예측한값이 맞을경우 리스트에 추가
    for n in range(0, len(y_test)):
        if yhat[n] == y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat) #예측한값이 맞는것을 그림으로 표현

    answer_list = [] #초기화
    # yhat[n] != y_test[n]일시 즉 예측한값이 틀릴경우 리스트에 추가
    for n in range(0, len(y_test)):
        if yhat[n] != y_test[n]:
            answer_list.append(n)

    draw_prediction(answer_list, 16,X_test,y_test,yhat) #예측한값이 틀린것을 그림으로 표현

def main():
    X_train, y_train, X_test, y_test = load_data() #데이터를 불러옴
    X_train, X_test = data_normalization(X_train,  X_test) #데이터 정규화

    model= makemodel(X_train, y_train, X_test, y_test,'glorot_uniform')#모델 생성

    baseline_history = model.fit(X_train,
                                 y_train,
                                 epochs=1,
                                 batch_size=512,
                                 validation_data=(X_test, y_test),
                                 verbose=2)
    # model.fit을 실행 (epochs는50회)
    evalmodel(X_test, y_test, model) #모델의 맞은값과 틀린값을 각각 그래프로 보겠다.
    plot_history([('baseline', baseline_history)]) # 그림표현

main() #메인
