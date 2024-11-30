import os

current_birates = [235, 375, 560, 750, 1050, 1750, 2350, 3000, 4300]

new_birates = [200, 300, 480, 750, 1200, 1850, 2850, 4300, 5300]


# list files in current directory
def run():
    current_directory = os.getcwd()

    print(current_directory)
    files = os.listdir(current_directory)
    for file in files:
        if os.path.isfile(file) and file != 'match_data.py':
            with open(file, 'r') as f:
                with open('./match_data/' + file, 'w') as new_f:
                    for line in f:
                            value = int(line.strip())
                            index = int(file.split('_')[2])
                            new_f.write(str(value * new_birates[index] // current_birates[index] ) + '\n')
        
                

run()