import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def parse_start_time(time_str):
    """ Convert a formatted string like '[2008. 4. 23. 6. 18. 19.921]' back to a list of floats. """
    time_str = time_str.strip("[]")
    time_list = list(map(float, time_str.split()))
    return time_list

def convert_to_datetime(time_list):
    """ Convert list of floats [year, month, day, hour, minute, second] into a datetime object. """
    base_time = datetime.datetime(
        int(time_list[0]), int(time_list[1]), int(time_list[2]),
        int(time_list[3]), int(time_list[4]), int(time_list[5])
    )

    fractional_seconds = time_list[5] - int(time_list[5])
    return base_time + datetime.timedelta(seconds=fractional_seconds)

def create_sequence(df, seq_length, target):
    """Convert time series data into sequences for LSTM."""
    # Input   Output
    # [1, 2, 3]   4
    # [2, 3, 4]   5
    # [3, 4, 5]   6
    df_as_np = df.to_numpy()
    sequences, labels = [], []

    target_index = df.columns.get_loc(target)

    for i in range(len(df_as_np) - seq_length):
        row = [r for r in df_as_np[i:i + seq_length]]
        sequences.append(row)

        label = df_as_np[i + seq_length, target_index]
        labels.append(label)

    return np.array(sequences), np.array(labels)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def create_sequences_autoencoder(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out):
        X.append(data[i:i+n_steps_in])
        y.append(data[i + n_steps_in: i + n_steps_in + n_steps_out, -1])
    return np.array(X), np.array(y)

def extract_features(df):
    """Extract features from a single cycle's time-series data"""
    features = {
        "Voltage_mean": np.mean(df["Voltage_measured"]),
        "Voltage_max": np.max(df["Voltage_measured"]),
        "Voltage_min": np.min(df["Voltage_measured"]),
        "Current_mean": np.mean(df["Current_measured"]),
        "Current_max": np.max(df["Current_measured"]),
        "Temperature_mean": np.mean(df["Temperature_measured"]),
        "Temperature_max": np.max(df["Temperature_measured"]),
        "Total_discharge_time": df["Time"].iloc[-1] - df["Time"].iloc[0]
    }

    return features

def extract_features_from_cycles(metadata, data_path):
    """
    Extracts features from all discharge cycles and merges them with discharge data.

    Parameters:
    - metadata (pd.DataFrame): DataFrame containing metadata (cycle_number, delta_time).
    - data_path (str): Directory where discharge cycle CSV files are stored.

    Returns:
    - pd.DataFrame: A DataFrame containing extracted features.
    """
    additional_features = []

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = f"{data_path}/{row['filename']}"
        df = pd.read_csv(file_path).copy()
        # Add filename and cycle number to the DataFrame
        features = extract_features(df)
        features["filename"] = row["filename"]
        features["cycle_number"] = row["cycle_number"]
        additional_features.append(features)

    return pd.DataFrame(additional_features)

def data_scaling(X, scaler=None):
    X_reshaped = X.reshape(-1, X.shape[-1])

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X_reshaped)
    else:
        X_scaled = scaler.transform(X_reshaped)

    X_scaled = X_scaled.reshape(X.shape)

    return X_scaled, scaler

def prepare_sequences_from_cycles(metadata, data_path, columns, seq_len=175):
    sequences = []

    scaler = MinMaxScaler(feature_range=(0, 1))

    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        file_path = f"{data_path}/{row['filename']}"
        df = pd.read_csv(file_path)[columns].copy()

        df[columns] = scaler.fit_transform(df[columns])

        if len(df) < seq_len:
            pad_len = seq_len - len(df)
            df = pd.concat([df, pd.DataFrame(np.zeros((pad_len, len(columns))), columns=columns)])
        else:
            df = df.iloc[:seq_len]

        sequences.append(df.values)

    return np.array(sequences)

def create_encoder_decoder_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps):
        input_seq = data[i : i + input_steps]
        output_seq = data[i + input_steps : i + input_steps + output_steps]
        X.append(input_seq)
        y.append(output_seq)
    return np.array(X), np.array(y)
