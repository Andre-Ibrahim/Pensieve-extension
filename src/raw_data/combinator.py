import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import os
import random
from tqdm import trange

def combine(new_file):
    script_dir = os.path.dirname(__file__)

    relative_paths = [
        './5G/time_dl_bandwidth.csv'
    ]

    file_paths = [os.path.join(script_dir, path) for path in relative_paths]

    data_frames = [pd.read_csv(file_path) for file_path in file_paths]

    data_frames = []
    for file_path in file_paths:
        try:
            network_type = os.path.basename(os.path.dirname(file_path)).upper()

            df = pd.read_csv(file_path)
            df['Network_Type'] = network_type
            data_frames.append(df)

            print(f"Loaded {file_path}:")
            print(df.head())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")


    for df in data_frames:
        print(df.head())

    output_file_path = os.path.join(script_dir, new_file)

    last_time = 0

    for _ in range(100):
        chosen_df = random.choice(data_frames)
        
        rows , _  = chosen_df.shape

        new_df = pd.DataFrame(columns=chosen_df.columns)

        random_row_number = random.randint(0, rows - 50)

        timeinterval = 0

        startrow = chosen_df.iloc[random_row_number]

        startrow_new = pd.DataFrame({'Network_Type': [startrow['Network_Type']], 'Time_Seconds': [last_time], 'DL_bitrate': [startrow['DL_bitrate']]})

        new_df = pd.concat([new_df, startrow_new], ignore_index=True)
        
        index = 1

        while timeinterval < 5:

            nextrow = chosen_df.iloc[random_row_number + index]
            new_time = 0
            
            new_time = last_time + (float(nextrow['Time_Seconds']) - float(startrow['Time_Seconds']))

            DL_bitrate = nextrow['DL_bitrate']
            network_type = nextrow['Network_Type']

            if(network_type == '5G' and DL_bitrate < 1.0):
                print("5G zero")
                DL_bitrate = 20.0
            
            if(network_type == '4G' and DL_bitrate < 0.0):
                print("3G zero")
                DL_bitrate = 3.0

            if(network_type == '3G' and DL_bitrate < 0.0):
                print("3G zero")
                DL_bitrate = 1.0

            nextrow_new = pd.DataFrame({'Network_Type': [startrow['Network_Type']], 'Time_Seconds': [new_time], 'DL_bitrate': [DL_bitrate]})
            
            index += 1

            new_df = pd.concat([new_df, nextrow_new], ignore_index=True)
            timeinterval += float(nextrow_new['Time_Seconds'].iloc[0]) - float(startrow_new['Time_Seconds'].iloc[0])

        last_time += timeinterval
        new_df.to_csv(output_file_path, mode='a', index=False, header=not os.path.exists(output_file_path))


if __name__ == "__main__":
    for i in trange(1, 800):
        combine(f'combined_data{i}.csv')