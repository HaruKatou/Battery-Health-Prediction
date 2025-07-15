import numpy
import pandas
import time
import tensorflow as tf
import random
import os

from keras.api import optimizers
from keras.api.utils import plot_model
from keras.api.layers import Dense, LSTM ,Dropout, SimpleRNN, Conv1D, MaxPooling1D
from keras_model import ATSLSTM
from keras.api.models import Sequential, load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib
from tqdm import trange
from math import sqrt

seed = 30

os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)
random.seed(seed)
numpy.random.seed(seed)


def load_dataset(datasource1: str, datasource2: str, datasource3: str, datasource4: str) -> (numpy.ndarray, MinMaxScaler):
    # 50 cycles of the first 3 batteries and whole cycles of the last battery are used
    dataframe1 = pandas.read_csv(datasource1, usecols=[1])
    dataframe1 = dataframe1.fillna(method='pad')
    dataset1 = dataframe1.values
    dataset1 = dataset1.astype('float32')
    dataset1 = dataset1[0:50]

    dataframe2 = pandas.read_csv(datasource2, usecols=[1])
    dataframe2 = dataframe2.fillna(method='pad')
    dataset2 = dataframe2.values
    dataset2 = dataset2.astype('float32')
    dataset2 = dataset2[0:50]

    dataframe3 = pandas.read_csv(datasource3, usecols=[1])
    dataframe3 = dataframe3.fillna(method='pad')
    dataset3 = dataframe3.values
    dataset3 = dataset3.astype('float32')
    dataset3 = dataset3[0:50]

    dataframe4 = pandas.read_csv(datasource4, usecols=[1])
    dataframe4 = dataframe4.fillna(method='pad')
    dataset4 = dataframe4.values
    dataset4 = dataset4.astype('float32')

    dataset = numpy.concatenate((dataset1, dataset2, dataset3, dataset4), axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, scaler


def create_dataset(dataset: numpy.ndarray, look_back: int=1) -> (numpy.ndarray, numpy.ndarray):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return numpy.array(data_x), numpy.array(data_y)


def build_model() -> Sequential:
    model = Sequential()
    model.add(Conv1D(filters=52, kernel_size=6, strides=3, padding='same', activation='relu',input_shape=(10, 3)))
    model.add(MaxPooling1D(pool_size=2, padding='valid'))
    model.add(ATSLSTM(23, stateful=False))
    model.add(Dropout(0.035))
    model.add(Dense(1))
    optimizer = optimizers.Adam(learning_rate=0.0012, beta_1=0.9, beta_2=0.999, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


def make_forecast(model: Sequential, look_back_buffer: numpy.ndarray, timesteps: int=1, batch_size: int=1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model.predict(look_back_buffer, batch_size)
        # add prediction to result
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        look_back_buffer = look_back_buffer.reshape(1, 30, 1)
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
        look_back_buffer = look_back_buffer.reshape(1, 10, 3)
    return forecast_predict

def make_forecast_until_EOL(model, look_back_buffer, scaler, capacity_threshold=1.4, batch_size=1):
    forecast_predict = numpy.empty((0, 1), dtype=numpy.float32)
    step = 0
    predicted_capacity = float('inf')

    while predicted_capacity > capacity_threshold and step < 300:  # Max cap to avoid infinite loop
        cur_predict = model.predict(look_back_buffer, batch_size=batch_size)
        forecast_predict = numpy.concatenate([forecast_predict, cur_predict], axis=0)
        predicted_capacity = scaler.inverse_transform(cur_predict)[0][0]
        step += 1

        # add new axis to prediction to make it suitable as input
        cur_predict = numpy.reshape(cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        look_back_buffer = look_back_buffer.reshape(1, 30, 1)
        # remove oldest prediction from buffer
        look_back_buffer = numpy.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = numpy.concatenate([look_back_buffer, cur_predict], axis=1)
        look_back_buffer = look_back_buffer.reshape(1, 10, 3)

    return numpy.array(forecast_predict), step  # predicted RUL


def main():
    datasource5 = r'./data/rul/5-capacity168.csv'
    datasource6 = r'./data/rul/6-capacity168.csv'
    datasource7 = r'./data/rul/7-capacity168.csv'
    datasource18 = r'./data/rul/18-capacity132.csv'

    dataset, scaler = load_dataset(datasource5, datasource6, datasource18, datasource7)
    joblib.dump(scaler, r'./result_cnn/scaler_rul.pickle')

    look_back = 30
    dataset_x, dataset_y = create_dataset(dataset, look_back)
    dataset_x = numpy.concatenate((dataset_x[0:20], dataset_x[50:70], dataset_x[100:120], dataset_x[150:]), axis=0)
    dataset_y = numpy.concatenate((dataset_y[0:20], dataset_y[50:70], dataset_y[100:120], dataset_y[150:]), axis=0)
    dataset_x = numpy.reshape(dataset_x, (dataset_x.shape[0], 10, 3))

    batch_size = 13
    loss_list = []
    starttime = time.time()
    model = build_model()
    for _ in trange(150, desc='fitting model\t', mininterval=1.0):
        print("dataset_x:", dataset_x.shape if dataset_x is not None else None)
        print("dataset_y:", dataset_y.shape if dataset_y is not None else None)
        history = model.fit(dataset_x, dataset_y, epochs=1, batch_size=batch_size, verbose=1, shuffle=False)
        plot_model(model, to_file=r'./result_cnn/rul_model_structure.png', show_shapes=True)
        loss_list.append(history.history['loss'][0])
        with open('./result/rul_loss.txt', 'a', encoding='utf-8') as f:
            f.write(str(history.history['loss'][0]) + "\n")
        for layer in model.layers:
            if hasattr(layer, 'reset_states'):
                layer.reset_states()
        # model.reset_states()
    model.save(r'./result_cnn/rul_model.h5')
    endtime = time.time()
    dtime = endtime - starttime

    # generate predictions for training
    dataset_predict = model.predict(dataset_x, batch_size)

    # generate forecast predictions
    # Take 31th to 50th cycles of B0005 as input
    # forecast_predict = make_forecast(model, dataset_x[19:20, :], timesteps=118, batch_size=batch_size)
    forecast_predict, predicted_rul = make_forecast_until_EOL(model, dataset_x[19:20, :], scaler, capacity_threshold=1.4, batch_size=batch_size)

    # invert dataset and predictions
    dataset = scaler.inverse_transform(dataset)
    dataset_predict = scaler.inverse_transform(dataset_predict)
    dataset_y = scaler.inverse_transform([dataset_y])
    forecast_predict = scaler.inverse_transform(forecast_predict)

    with open(r'./result_cnn/rul_prediction_data' + ".txt", 'a', encoding='utf-8') as f:
        for m in range(len(forecast_predict)):
            f.write(str(forecast_predict[m]) + "\n")
    print("程序运行时间：%.8s s" % dtime)
    index = []

    dataset_score = sqrt(mean_squared_error(dataset_y[0], dataset_predict[:, 0]))
    print('Train Dataset Score: %.4f RMSE' % dataset_score)
    forecast_score = sqrt(mean_squared_error(dataset_y[0, -118:], forecast_predict[:118, 0]))
    print('Test Dataset Score: %.4f RMSE' % forecast_score)
    rul_error = abs(76 - predicted_rul)
    print('Predicted RUL: %d, Error: %d' % (predicted_rul, rul_error))
    index.append('train_dataset_score: %.5f' % dataset_score)
    index.append('test_dataset_score: %.5f' % forecast_score)
    index.append('rul_error: %d' % rul_error)
    index.append('time: %1f' % dtime)

    with open(r'./result_cnn/soh_prediction_result_#7_50.txt', 'a', encoding='utf-8') as f:
        for j in range(len(index)):
            f.write(str(index[j]) + "\n")


if __name__ == '__main__':
    main()