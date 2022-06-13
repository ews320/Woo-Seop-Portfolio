import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Keras에서 필요한 항목 가져오기
from keras.models import Sequential
from keras.layers import Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.utils import plot_model


def create_model(input_shape, pool_size):
    # 여기에서 실제 신경망을 만듭니다.
    model = Sequential()
    # 들어오는 입력을 정규화합니다. 첫 번째 레이어가 작동하려면 입력 모양이 필요합니다.
    model.add(BatchNormalization(input_shape=input_shape))

    # 모델 요약을 더 쉽게 읽을 수 있도록 아래 레이어의 이름이 변경되었습니다. 이것은 필요하지 않습니다
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv1'))

    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv2'))

    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))

    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))

    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))

    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))

    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))

    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Conv7'))
    model.add(Dropout(0.2))

    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))

    # Upsample 1
    model.add(UpSampling2D(size=pool_size))

    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))

    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))

    # Upsample 2
    model.add(UpSampling2D(size=pool_size))

    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))

    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))

    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))

    # Upsample 3
    model.add(UpSampling2D(size=pool_size))

    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Deconv6'))

    # 최종 레이어 - 하나의 채널만 포함하므로 1 필터
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1, 1), activation='relu', name='Final'))

    return model


def main():
    # 훈련 이미지 로드
    train_images = pickle.load(open("full_CNN_train.p", "rb"))

    # 이미지 라벨 로드
    labels = pickle.load(open("full_CNN_labels.p", "rb"))

    # 신경망이 원하는 대로 배열로 만듭니다.
    train_images = np.array(train_images)
    labels = np.array(labels)

    # 레이블 정규화 - 훈련 이미지가 네트워크에서 시작하도록 정규화됩니다.
    labels = labels / 255

    # 레이블과 함께 이미지를 섞은 다음 훈련/검증 세트로 분할
    train_images, labels = shuffle(train_images, labels)
    # 테스트 크기는 10% 또는 20%일 수 있습니다.
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    # 아래의 배치 크기, 에포크 및 풀 크기는 모두 최적화를 위해 조정해야 하는 매개변수입니다.
    batch_size = 128
    epochs = 20  # 10 to 20
    pool_size = (2, 2)
    input_shape = X_train.shape[1:]

    # 신경망 만들기
    model = create_model(input_shape, pool_size)

    # 모델이 더 적은 데이터를 사용하도록 돕기 위해 생성기를 사용
    # 채널 이동은 그림자에 약간 도움이 됩니다.
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)

    # 모델 컴파일 및 학습
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=epochs, verbose=1, validation_data=(X_val, y_val), callbacks=[early_stop])

    # 훈련이 완료된 후 레이어 고정
    model.trainable = False
    model.compile(optimizer='Adam', loss='mse', metrics=['mae'])

    # 모델 아키텍처 및 가중치 저장
    model.save('full_CNN_model.h5')
    plot_model(model, to_file='model_shapes.png')

    plt.plot(history.history['mae'], label='mae')
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Accuracy and Loss History')
    plt.xlabel('epoch')
    plt.legend(loc='center right')
    plt.savefig('result.png')

    # 모델 요약 표시
    model.summary()

if __name__ == '__main__':
    main()
