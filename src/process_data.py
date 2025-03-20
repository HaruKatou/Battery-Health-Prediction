import pandas as pd
import numpy as np
from tqdm import tqdm

def compute_battery_health(metadata_file, data_folder, output_path, num_bins=20, nominal_capacity=2.0, min_capacity=1.4, disc_cutoff_voltage=2.7):
    """
    Estimates the State of Health (SoH) of battery discharge cycle.

    Parameters:
    - metadata_path: Path to the metadata CSV file.
    - data_folder: Folder containing cycle data CSV files.
    - output_path: Path to save the processed SoH dataset.
    - num_bins: Number of bins for downsampling each cycle.
    - nominal_capacity: Nominal capacity of the battery in Ah.
    - min_capacity: Minimum capacity threshold for valid cycles.
    - discharge_end_voltage: The voltage level at which the discharge cycle is considered complete.
    """
    metadata = pd.read_csv(metadata_file)

    # Get discharge cycles
    metadata['battery_id'] = metadata['battery_id'].astype(str)
    discharge_metadata = metadata[metadata['type'] == 'discharge'].copy()
    discharge_metadata['cycle_number'] = discharge_metadata.groupby('battery_id').cumcount() + 1

    processed_dfs = []

    for _, row in tqdm(discharge_metadata.iterrows(), total=len(discharge_metadata)):
        try:
            file_path = f"{data_folder}/{row['filename']}"
            df = pd.read_csv(file_path).copy()

            cutoff_idx = df[df['Voltage_measured'] < disc_cutoff_voltage].index.min()
            trimmed_df = df if pd.isna(cutoff_idx) else df.iloc[:cutoff_idx].copy()

            # Calculate capacity
            trimmed_df['Time_diff_hr'] = trimmed_df['Time'].diff().fillna(0) / 3600
            # delta Q = I * delta t
            trimmed_df['Delta_Q'] = trimmed_df['Current_measured'] * trimmed_df['Time_diff_hr']
            capacity = abs(trimmed_df['Delta_Q'].sum())

            if capacity > min_capacity:
                trimmed_df['battery_id'] = row['battery_id']
                trimmed_df['cycle_number'] = row['cycle_number']

                # Calculate SoC
                trimmed_df['Cumulative_Q'] = trimmed_df['Delta_Q'].cumsum()
                trimmed_df['SoC'] = (1 + trimmed_df['Cumulative_Q'] / capacity) * 100

                # Calculate SoH (nominal capacity = 2Ah)
                trimmed_df['SoH'] = (capacity / nominal_capacity) * 100

                # Downsample each cycle to exactly num_bins rows by averaging
                bins = np.array_split(trimmed_df, num_bins)
                agg_rows = []
                for b in bins:
                    if b.empty:
                        continue
                    agg_rows.append({
                        'Voltage_measured': b['Voltage_measured'].mean(),
                        'Current_measured': b['Current_measured'].mean(),
                        'Temperature_measured': b['Temperature_measured'].mean(),
                        'SoC': b['SoC'].mean(),
                        'cycle_number': b['cycle_number'].iloc[0],
                        'battery_id': b['battery_id'].iloc[0],
                        'SoH': b['SoH'].iloc[0]
                    })

                cycle_df = pd.DataFrame(agg_rows)
                if len(cycle_df) == num_bins:
                    processed_dfs.append(cycle_df)

        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")

    if processed_dfs:
        full_dataset = pd.concat(processed_dfs)
        full_dataset.to_csv(output_path, index=False)
        print(f"Saved processed dataset to {output_path}")
    else:
        print("No valid files found!")

compute_battery_health(
    metadata_file="dataset/processed/metadata.csv",
    data_folder="dataset/processed/data",
    output_path="dataset/processed/battery_health_dataset.csv"
)