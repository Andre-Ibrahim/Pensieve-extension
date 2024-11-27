import os
from pathlib import Path

def combine_logs(input_directory, output_file):
    current_directory = Path(input_directory)
    combined_data = []
    last_time = 0

    for log_file in sorted(current_directory.iterdir()):
        if log_file.is_file() and log_file.suffix == '.log':
            with open(log_file, 'r') as f:
                for line in f:
                    time, value = map(float, line.split())
                    adjusted_time = time + last_time + 1
                    combined_data.append(f"{adjusted_time},{value}\n")
                last_time = adjusted_time

    with open(output_file, 'w') as out_f:
        out_f.writelines("Time_Seconds,DL_bitrate\n")
        out_f.writelines(combined_data)

if __name__ == "__main__":
    input_directory = os.getcwd()
    output_file = '3G_logs2.csv'
    combine_logs(input_directory, output_file)