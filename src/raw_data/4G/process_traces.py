from pathlib import Path
import pandas as pd
from datetime import datetime

current_directory = Path.cwd()

def parse_timestamp(timestamp_str):
    return datetime.strptime(timestamp_str, "%Y.%m.%d_%H.%M.%S")

# Initialize cumulative end time in seconds
LastFileEndTime = 0

output_file = '4G_time_bandwidth.csv'

for i, dir_path in enumerate(current_directory.iterdir()):
    if dir_path.is_dir():
        for j, file_path in enumerate(dir_path.iterdir()):
            if file_path.is_file():
                df = pd.read_csv(file_path)

                start_time = pd.Timestamp(parse_timestamp(df['Timestamp'].iloc[0]))

                df['Timestamp'] = df['Timestamp'].apply(parse_timestamp)

                df['Time_Seconds'] = (df['Timestamp'] - start_time).dt.total_seconds() + LastFileEndTime

                time_dl_df = df[['Time_Seconds', 'DL_bitrate']].copy()

                time_dl_df['DL_bitrate'] = time_dl_df['DL_bitrate'] / 1000

                LastFileEndTime = df['Time_Seconds'].iloc[-1]

                if Path(output_file).exists():
                    time_dl_df.to_csv(output_file, mode='a', index=False, header=False)
                else:
                    time_dl_df.to_csv(output_file, mode='w', index=False, header=True)
