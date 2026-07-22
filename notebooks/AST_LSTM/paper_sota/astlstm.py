import numpy as np
import pandas as pd
import time
import tensorflow as tf
import random
import os
import sys
sys.path.append(os.path.abspath("../../"))
from AST_LSTM.paper_sota.keras_model import ATSLSTM

from keras import optimizers
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt

CONFIG = {
    'SEED': 42,
    'SP_VALUES': [50, 70, 90],
    'LOOK_BACK': 30,
    'BLOCKS_NUM': 84,
    'LR': 2.6e-3,
    'BATCH_SIZE': 17,
    'EPOCHS': 170,
    'DROPOUT_RATE': 1.1e-2,
    'TIMESTEPS': 168,
    'THRESHOLD': 1.4,
    'MAX_PREDICTION_CYCLES': 300,
    'DATASOURCE6': r'./data/rul/6-capacity168.csv',
    'DATASOURCE18': r'./data/rul/18-capacity132.csv',
    'DATASOURCE5': r'./data/rul/5-capacity168.csv'
}

def set_seed(seed=42):
    """
    Set seed for reproducibility
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_dataset_2(datasource1: str, datasource2: str) -> (np.ndarray, MinMaxScaler):
    dataframe1 = pd.read_csv(datasource1, usecols=[1])
    dataset1 = dataframe1.values.astype('float32')

    dataframe2 = pd.read_csv(datasource2, usecols=[1])
    dataset2 = dataframe2.values.astype('float32')

    dataset = np.concatenate((dataset1, dataset2), axis=0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    return dataset, scaler

def load_dataset_3(datasource1: str) -> (np.ndarray, MinMaxScaler):
    dataframe1 = pd.read_csv(datasource1, usecols=[1])
    dataset1 = dataframe1.values.astype('float32')

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset1)

    return dataset, scaler

def create_dataset(dataset: np.ndarray, look_back: int=1) -> (np.ndarray, np.ndarray):
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i : (i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)

def build_model(input_shape, blocks_num=84, dropout_rate=1.1e-2, lr=2.6e-3) -> Sequential:
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(ATSLSTM(blocks_num, stateful=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

def make_forecast_until_EOL(model, look_back_buffer, scaler, capacity_threshold=1.4, batch_size=1):
    forecast_predict = np.empty((0, 1), dtype=np.float32)
    step = 0
    predicted_capacity = float('inf')

    while predicted_capacity > capacity_threshold and step < CONFIG['MAX_PREDICTION_CYCLES']:
        cur_predict = model.predict(look_back_buffer, batch_size=batch_size, verbose=0)
        forecast_predict = np.concatenate([forecast_predict, cur_predict], axis=0)
        predicted_capacity = scaler.inverse_transform(cur_predict)[0][0]
        step += 1

        # Prepare next input
        cur_predict = cur_predict.reshape(1, 1, 1)
        look_back_buffer = look_back_buffer.reshape(1, CONFIG['LOOK_BACK'], 1)
        look_back_buffer = np.concatenate([look_back_buffer[:, 1:, :], cur_predict], axis=1)

    return np.array(forecast_predict), step

def main():
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    set_seed(CONFIG['SEED'])

    dataset, scaler = load_dataset_2(CONFIG['DATASOURCE6'], CONFIG['DATASOURCE18'])
    test_dataset, test_scaler = load_dataset_3(CONFIG['DATASOURCE5'])

    battery_splits = {
        "B0005": 168,
        "B0018": 132
    }

    X_seqs, y_seqs = [], []
    start = 0
    for battery_id, num_cycles in battery_splits.items():
        end = start + num_cycles
        battery_data = dataset[start:end]

        X_battery, y_battery = create_dataset(battery_data, CONFIG['LOOK_BACK'])

        X_seqs.append(X_battery)
        y_seqs.append(y_battery)
        start = end

    dataset_x = np.concatenate(X_seqs, axis=0)
    dataset_y = np.concatenate(y_seqs, axis=0)

    dataset_x = np.expand_dims(dataset_x, axis=-1)
    test_x, test_y = create_dataset(test_dataset, CONFIG['LOOK_BACK'])
    test_x = np.expand_dims(test_x, axis=-1)

    df_b0005 = pd.read_csv(CONFIG['DATASOURCE5'], usecols=[1]).fillna(method='pad')
    true_b0005 = df_b0005.values.astype('float32').flatten()

    model = build_model(input_shape=(CONFIG['LOOK_BACK'], 1), blocks_num=CONFIG['BLOCKS_NUM'], dropout_rate=CONFIG['DROPOUT_RATE'], lr=CONFIG['LR'])

    callbacks = [
        EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
    ]

    model.fit(dataset_x, dataset_y, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'], verbose=1, shuffle=False, callbacks=callbacks)

    seed_results = []
    for sp in CONFIG['SP_VALUES']:
        true_b0005_sp = true_b0005[sp:]

        sample_num = sp - CONFIG['LOOK_BACK']
        look_back_buffer = test_x[sample_num:sample_num+1]
        forecast_predict, predicted_rul = make_forecast_until_EOL(model, look_back_buffer, scaler=scaler,
                                                                  capacity_threshold=CONFIG['THRESHOLD'], batch_size=CONFIG['BATCH_SIZE'])

        forecast_predict = scaler.inverse_transform(forecast_predict)

        compare_timesteps = min(CONFIG['TIMESTEPS'] - sp, predicted_rul)
        rmse = sqrt(mean_squared_error(true_b0005_sp[:compare_timesteps], forecast_predict[:compare_timesteps, 0]))

        true_rul = 124 - sp
        ae = abs(predicted_rul - true_rul)

        seed_results.append({'SP': sp, 'RMSE': rmse, 'AE': ae, 'Predicted RUL': predicted_rul})

    # Print results for each SP
    print(f"\nResults for seed {CONFIG['SEED']}:")
    for result in seed_results:
        print(f"SP={result['SP']}: RMSE={result['RMSE']:.4f}, AE={result['AE']}, Predicted RUL={result['Predicted RUL']}")

    # Compute and print averages
    avg_rmse = np.mean([result['RMSE'] for result in seed_results])
    avg_ae = np.mean([result['AE'] for result in seed_results])
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    print(f"Average AE: {avg_ae:.4f}")

if __name__ == "__main__":
    main()