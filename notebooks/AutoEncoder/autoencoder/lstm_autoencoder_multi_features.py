import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense, Dropout, InputLayer
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import RootMeanSquaredError
import sys
sys.path.append(os.path.abspath("../../"))
from LSTM.preprocessing import prepare_sequences_from_cycles, create_sequences
from AST_LSTM.paper_sota.keras_model import ATSLSTM
from keras.layers import Layer

strategy = tf.distribute.MirroredStrategy()
print(f"Number of devices: {strategy.num_replicas_in_sync}")

CONFIG = {
    'SEED': 42,
    'SP': [50, 60, 70],
    'LOOK_BACK': 30,
    'LATENT_DIM': 11,
    'TIMESTEPS': 168,
    'EPOCHS_AUTOENCODER': 10,
    'EPOCHS_MODEL': 100,
    'BATCH_SIZE_AUTOENCODER': 16,
    'BATCH_SIZE_MODEL': 32,
    'PATIENCE': 5,
    'THRESHOLD': 1.4,
    'MAX_PREDICTION_CYCLES': 300,
    'FEATURES': ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'Current_load', 'Voltage_load'],
    'DATA_PATH': "../../../dataset/processed/data",
    'DATASET_PATH': "../../../dataset/processed/preprocessed_discharge_data.csv",
    'MODEL_PATH': 'models/model.keras',
    'AUTOENCODER_PATH': 'autoencoder/best_autoencoder.keras'
}

def set_seed(seed):
    """Set seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_and_prepare_data(dataset_path, battery_ids, features, data_path):
    """Load and prepare sequences for specified batteries."""
    dataset = pd.read_csv(dataset_path)
    sequences = []
    capacities = []

    for battery_id in battery_ids:
        battery_data = dataset[dataset["battery_id"] == battery_id].copy()
        seq = prepare_sequences_from_cycles(battery_data, data_path, features)
        sequences.append(np.array(seq))
        capacities.append(battery_data["Capacity"])

    X = np.concatenate(sequences, axis=0)
    capacity = pd.concat(capacities, axis=0).reset_index(drop=True)
    return X, capacity

def build_autoencoder(timesteps, features, latent_dim):
    """Build and compile the autoencoder and encoder models."""
    input_layer = Input(shape=(timesteps, features))
    encoded = LSTM(64, activation='sigmoid', return_sequences=True)(input_layer)
    encoded = LSTM(32, activation='sigmoid', return_sequences=False)(encoded)
    encoded = Dense(latent_dim, activation='relu', name='bottleneck')(encoded)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(32, activation='sigmoid', return_sequences=True)(decoded)
    decoded = LSTM(64, activation='sigmoid', return_sequences=True)(decoded)
    decoded = TimeDistributed(Dense(features, activation='linear'))(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(loss="mse", optimizer=Adam(), metrics=[RootMeanSquaredError()])
    return autoencoder, encoder

def train_autoencoder(autoencoder, X, epochs, batch_size, model_path, patience):
    """Train the autoencoder model."""
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]
    autoencoder.fit(
        X, X,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

def build_model(timesteps, n_features):
    model = Sequential([
        InputLayer((timesteps, n_features)),
        LSTM(64, return_sequences=True, activation='tanh'),
        LSTM(32, return_sequences=True, activation='tanh'),
        LSTM(16, return_sequences=False, activation='tanh'),
        Dense(16, activation='relu'),
        Dense(n_features, activation='linear')
    ])
    model.compile(loss="mse", optimizer=Adam(), metrics=[RootMeanSquaredError()])
    return model

def build_model_2(timesteps, n_features):
    model = Sequential([
        InputLayer((timesteps, n_features)),
        ATSLSTM(64, return_sequences=True, activation='tanh'),
        LSTM(32, return_sequences=True, activation='tanh'),
        LSTM(16, return_sequences=False, activation='tanh'),
        Dense(16, activation='relu'),
        Dense(n_features, activation='linear')
    ])
    model.compile(loss="mse", optimizer=Adam(), metrics=[RootMeanSquaredError()])
    return model

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = Dense(1, activation='tanh')

    def call(self, inputs):
        e = self.score_dense(inputs)                # (B, T, 1)
        e = tf.squeeze(e, axis=-1)                  # (B, T)
        a = tf.nn.softmax(e, axis=1)                 # (B, T)
        a = tf.expand_dims(a, axis=-1)               # (B, T, 1)
        weighted = inputs * a                        # (B, T, F)
        return tf.reduce_sum(weighted, axis=1)       # (B, F)
    
def build_palstm_model(timesteps, n_features):
    inputs = Input(shape=(timesteps, n_features))
    lstm_out = LSTM(64, activation='tanh', return_sequences=True)(inputs)
    attention_out = AttentionLayer()(lstm_out)
    output = Dense(1, activation='linear')(attention_out)
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss="mse", optimizer=SGD(learning_rate=0.02), metrics=[RootMeanSquaredError()])
    return model

def train_model(model, X, y, epochs, batch_size, model_path, patience):
    """Train the ATS-LSTM model."""
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss')
    ]
    model.fit(
        X, y,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

def predict_rul(model, initial_input, capacity_scaler, latent_dim, threshold, max_cycles):
    """Predict RUL until capacity reaches threshold."""
    predicted_capacities = []
    current_input = initial_input.copy()
    i = 0

    while i < max_cycles:
        y_pred = model.predict(current_input, verbose=0)
        capacity_pred = y_pred[0, -1]
        capacity_pred_real = capacity_scaler.inverse_transform([[capacity_pred]])[0, 0]
        predicted_capacities.append(capacity_pred_real)

        if capacity_pred_real <= threshold:
            break

        next_step = np.zeros((1, 1, latent_dim + 1))
        next_step[0, 0, :] = y_pred[0, :]
        current_input = np.concatenate([current_input[:, 1:, :], next_step], axis=1)
        i += 1

    return predicted_capacities, i

def evaluate_model(discharge_data, encoder, model, sp, look_back, latent_dim, threshold, max_cycles, timesteps, features, data_path, capacity_scaler):
    """Evaluate the model for a given starting point."""
    X = prepare_sequences_from_cycles(discharge_data, data_path, features)
    encoded_features = encoder.predict(X)

    capacity = np.array(discharge_data["Capacity"])
    capacity_scaled = capacity_scaler.transform(capacity.reshape(-1, 1)).flatten()

    combined_features = np.hstack((encoded_features, capacity_scaled.reshape(-1, 1)))

    X_seq, y_seq = create_sequences(combined_features, look_back)

    sample_num = sp - look_back
    initial_input = X_seq[sample_num-1:sample_num]
    X_input = initial_input.reshape(1, look_back, latent_dim + 1)

    predicted_capacities, predicted_rul = predict_rul(
        model, X_input, capacity_scaler, latent_dim, threshold, max_cycles
    )

    compare_timesteps = min(timesteps - sp, predicted_rul)
    true_capacities = discharge_data["Capacity"].values[sp:sp + compare_timesteps]
    predicted_capacities = predicted_capacities[:compare_timesteps]

    rmse = np.sqrt(mean_squared_error(true_capacities, predicted_capacities))
    # 124, 108, 96
    true_rul = 96 - sp
    ae = abs(predicted_rul - true_rul)

    return rmse, ae, predicted_rul

def main():
    set_seed(CONFIG['SEED'])
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    # Load and prepare training data
    X_train, capacity_train = load_and_prepare_data(
        CONFIG['DATASET_PATH'],
        ["B0005", "B0018"],
        CONFIG['FEATURES'],
        CONFIG['DATA_PATH']
    )

    timesteps, features = X_train.shape[1], X_train.shape[2]

    # Build and train autoencoder
    autoencoder, encoder = build_autoencoder(timesteps, features, CONFIG['LATENT_DIM'])
    train_autoencoder(
        autoencoder, X_train,
        CONFIG['EPOCHS_AUTOENCODER'],
        CONFIG['BATCH_SIZE_AUTOENCODER'],
        CONFIG['AUTOENCODER_PATH'],
        CONFIG['PATIENCE']
    )

    encoded_features = encoder.predict(X_train)
    capacity_scaler = MinMaxScaler()
    capacity_scaled = capacity_scaler.fit_transform(np.array(capacity_train).reshape(-1, 1)).flatten()
    combined_features = np.hstack((encoded_features, capacity_scaled.reshape(-1, 1)))

    # X_seq, y_seq = create_sequences(combined_features, CONFIG['LOOK_BACK'])

    features_b0006 = combined_features[:168]
    features_b0018 = combined_features[168:]

    X_b0006, y_b0006 = create_sequences(features_b0006, CONFIG['LOOK_BACK'])
    X_b0018, y_b0018 = create_sequences(features_b0018, CONFIG['LOOK_BACK'])

    X_seq = np.concatenate([X_b0006, X_b0018], axis=0)
    y_seq = np.concatenate([y_b0006, y_b0018], axis=0)

    model = build_palstm_model(X_seq.shape[1], X_seq.shape[2])
    train_model(
        model, X_seq, y_seq,
        CONFIG['EPOCHS_MODEL'],
        CONFIG['BATCH_SIZE_MODEL'],
        CONFIG['MODEL_PATH'],
        CONFIG['PATIENCE']
    )

    dataset = pd.read_csv(CONFIG['DATASET_PATH'])
    discharge_data_b0005 = dataset[dataset["battery_id"] == "B0006"].copy()

    results = []
    for sp in CONFIG['SP']:
        rmse, ae, predicted_rul = evaluate_model(
            discharge_data_b0005, encoder, model,
            sp, CONFIG['LOOK_BACK'], CONFIG['LATENT_DIM'],
            CONFIG['THRESHOLD'], CONFIG['MAX_PREDICTION_CYCLES'],
            CONFIG['TIMESTEPS'], CONFIG['FEATURES'], CONFIG['DATA_PATH'], capacity_scaler
        )
        results.append({'SP': sp, 'RMSE': rmse, 'AE': ae, 'Predicted RUL': predicted_rul})

    # Print results
    print("\nEvaluation Results:")
    for result in results:
        print(f"SP={result['SP']}: RMSE={result['RMSE']:.4f}, AE={result['AE']}, Predicted RUL={result['Predicted RUL']}")

    # Compute and print averages
    avg_rmse = np.mean([result['RMSE'] for result in results])
    avg_ae = np.mean([result['AE'] for result in results])
    print(f"\nAverage RMSE: {avg_rmse:.4f}")
    print(f"Average AE: {avg_ae:.4f}")

if __name__ == "__main__":
    main()