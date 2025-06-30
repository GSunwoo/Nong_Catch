import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

def main():
    data = np.load('trainData/str_train_data_clprod.npz')
    # np.savez() 함수로 저장시 이미 데이터를 훈련용, 테스트용으로 분리했음
    X_train = data['X_train']
    X_test = data['X_test']
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = model_train(X_train, Y_train)
    model_eval(model, X_test, Y_test)

def build_model(in_shape):
    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=in_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # 이진 분류 출력 (0~1 확률)

    # 모델 컴파일
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    return model

def model_train(X, Y):
    model = build_model(X.shape[1:])

    model.fit(X,Y, batch_size=32, epochs=100)

    hdf5_file = './trainedModel/str_clprod_model.hdf5'
    model.save_weights(hdf5_file)
    return model

def model_eval(model, X, Y):
    score = model.evaluate(X, Y)
    print('loss =', score[0])
    print('accuracy =', score[1])

if __name__ == '__main__':
    main()