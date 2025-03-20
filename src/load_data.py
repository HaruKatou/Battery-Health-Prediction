import os
import pandas as pd
import numpy as np
import scipy.io

def format_matlab_time(matlab_time):
    """
    Convert MATLAB-style date vector into formatted string.
    """
    try:
        if isinstance(matlab_time, np.ndarray):
            matlab_time = matlab_time.tolist()

        if not isinstance(matlab_time, list) or len(matlab_time) != 6:
            return "[]"

        return f"[{matlab_time[0]:.0f}. {matlab_time[1]:.0f}. {matlab_time[2]:.0f}. " \
               f"{matlab_time[3]:.0f}. {matlab_time[4]:.0f}. {matlab_time[5]:.3f}]"

    except Exception:
        return "[]"

def process_cycle_data(cycle, cycle_type, file_path):
    """
    Saves the time-series data from a cycle to a CSV file.
    """
    if 'data' not in cycle.dtype.names:
        print(f"Skipping cycle, no 'data' field found: {file_path}")
        return

    data = cycle['data']

    charge_fields = ["Voltage_measured", "Current_measured", "Temperature_measured",
                     "Current_charge", "Voltage_charge", "Time"]
    discharge_fields = ["Voltage_measured", "Current_measured", "Temperature_measured",
                     "Current_load", "Voltage_load", "Time"]
    impedance_fields = ["Sense_current", "Battery_current", "Current_ratio",
                        "Battery_impedance", "Rectified_Impedance"]

    if cycle_type == 'charge':
        fields = charge_fields
    elif cycle_type == 'discharge':
        fields = discharge_fields
    elif cycle_type == 'impedance':
        fields = impedance_fields
    else:
        print(f"Skipping cycle, unknown cycle type: {file_path}")
        return

    max_len = 0
    data_dict = {}

    for name in fields:
        if name in data.dtype.names:
            arr = data[0][0][name].flatten()
            arr = arr.astype(object)
            data_dict[name] = arr
            max_len = max(max_len, len(arr))
        else:
            data_dict[name] = np.array([], dtype=object)

    for name in data_dict:
        cur_len = len(data_dict[name])
        if cur_len < max_len:
            data_dict[name] = np.concatenate(
                (data_dict[name], np.full((max_len - cur_len,), "", dtype=object))
            )

    df = pd.DataFrame(data_dict)
    df.to_csv(file_path, index=False)
    print(f"Saved {cycle_type} data to {file_path}")

file_id = 1

def process_metadata_and_cycles(cycle_data, battery_id, output_folder="dataset/processed/"):
    """
    Extract metadata and save both metadata and cycle time-series data into CSV files.
    - Metadata stored in `processed/metadata.csv`
    - Each cycle data stored in `processed/data/00001.csv`, `processed/data/00002.csv`, etc.
    """
    global file_id

    data_folder = os.path.join(output_folder, "data")
    os.makedirs(data_folder, exist_ok=True)

    cycle_info = []
    test_id = 0

    for i in range(cycle_data.shape[1]):
        cycle = cycle_data[0, i]
        cycle_type = cycle['type'][0]
        start_time = format_matlab_time(cycle["time"][0])
        ambient_temperature = cycle["ambient_temperature"][0][0]

        file_number = f"{file_id:05d}.csv"
        file_path = os.path.join(data_folder, file_number)

        process_cycle_data(cycle, cycle_type, file_path)

        capacity = cycle["data"]["Capacity"][0, 0].flatten()[0] if cycle_type == "discharge" and "Capacity" in cycle["data"].dtype.names else ""
        re = cycle["data"]["Re"][0, 0].flatten()[0] if cycle_type == "impedance" and "Re" in cycle["data"].dtype.names else ""
        rct = cycle["data"]["Rct"][0, 0].flatten()[0] if cycle_type == "impedance" and "Rct" in cycle["data"].dtype.names else ""

        cycle_info.append([
            cycle_type, start_time, ambient_temperature, battery_id, test_id, file_id, file_number, capacity, re, rct
        ])

        test_id += 1
        file_id += 1

    metadata_path = os.path.join(output_folder, "metadata.csv")

    metadata_df = pd.DataFrame(cycle_info, columns=[
        "type", "start_time", "ambient_temperature", "battery_id",
        "test_id", "uid", "filename", "Capacity", "Re", "Rct"
    ])
    metadata_df.to_csv(metadata_path, index=False, mode='a', header=not os.path.exists(metadata_path))
    print(f"Metadata saved to {metadata_path}")

dataset_folder = "dataset/raw"
output_folder = "dataset/processed"

if not os.path.exists(dataset_folder):
    print(f"Error: Folder '{dataset_folder}' not found.")
else:
    for filename in sorted(os.listdir(dataset_folder)):
        # Skip experiments with crashed software
        if filename.endswith(".mat") and filename not in {"B0049.mat", "B0050.mat", "B0051.mat", "B0052.mat"}:
            mat_path = os.path.join(dataset_folder, filename)
            battery_id = filename.split(".")[0]

            print(f"\nProcessing {filename} (Battery ID: {battery_id})")
            mat_data = scipy.io.loadmat(mat_path)[battery_id]
            cycle_data = mat_data[0, 0]['cycle']

            process_metadata_and_cycles(cycle_data, battery_id, output_folder)

    print("\nProcessing complete! Check the 'processed' folder for results.")
