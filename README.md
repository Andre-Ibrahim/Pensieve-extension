# This is a fork of Pensieve PPO [Pensieve-PPO](https://github.com/godka/Pensieve-PPO)


## Before running the project you need to setup the enivroment here are the suggested steps:

### create a python virtual environment

`python3 -m venv venv`

`source venv/bin/activate`


### Install requirements

`pip3 -r install requirements.txt`

### Before train your own model there are several steps you need to take to ensure you are training correctly

### 1. Data prep

#### 1.1 video segments

- make sure that VIDEO_BIT_RATE has the right value for your video segment sizes

- the video segment sizes should be a folder containing files of named video_size_{n} where n is the bitrate level for example video_size_6

- The data should be located under ./src/video_data

[Video data](https://github.com/godka/comyco-video-description-dataset)

#### 1.2 network traces

- Network traces are a collection of csv files with these collumns Time_Seconds,DL_bitrate,Network_Type

sample: 

Time_Seconds,DL_bitrate,Network_Type
0,0.765554716981,3G
1.0010000000002037,0.929310689311,3G
2.0210000000006403,0.930862745098,3G
3.0310000000008586,1.18108514851,3G
6.053000000001703,8.455,4G
8.053000000001703,13.856,4G
9.053000000001703,16.899,4G
11.053000000001703,1.07322343595,3G
12.07300000000214,1.53430588235,3G
13.085000000004584,1.34545454545,3G
14.08900000000176,1.01373705179,3G
17.14100000000508,3.064,4G
18.14100000000508,2.874,4G
20.14100000000508,6.341,4G
21.14100000000508,5.468,4G
25.14100000000508,0.421,5G
26.14100000000508,1.371,5G

- for this experiment you can create as many csv files provided they are at minimum the size of the video provided.

the .csv files are in src/test_heterogenous and src/train_heterogenous

[3G Dataset](http://skuld.cs.umass.edu/traces/mmsys/2013/pathbandwidth/)

[4G Dataset](https://users.ugent.be/~jvdrhoof/dataset-4g/)

[5G Dataset](https://github.com/uccmisl/5Gdataset/tree/master?tab=readme-ov-file)

### 2. configure training parameters

have a look at sr/train.py make sure the parameters passed make sense for your use cases

have a look at rewardFunctions, the one designed and selected for this project is `reward5` you might want to design your own.

*** Disclaimer: the reward functions designed are experimental and could not work as intended

Finally before running the command update NAME in train.py so that you can reference your trained model later.

### 3. Its time to train!! 

run this command
(you might need to run it with sudo)

`python3 train.py`

you will be presented with a tqdm loading bar which will estimate completion time after the first epoch

after training you might want to visualize your results first you will need to test your model here how that could look like

### 4. For testing

run this command (possibly need sudo here as well)

`python3 test.py ppo/heterogenous_switch_rate_tuned/nn_model_ep_80000.pth 0.222 2 3  80000 testingforoutput`

where  
- `ppo/heterogenous_switch_rate_tuned/nn_model_ep_80000.pth` is the path to your model

- `0.222` is hyperparameter alpha

- `2` is hyperparameter beta

- `3` is hyperparameter gamma

- `testingforoutput` is the name of the scheme that will be created

### 5. Results

Now that you ran the test you can add your new scheme `testingforoutput` in `SCHEMES` and run

`python3 plot.py`

you should see your plot in /src with the names defined in plot.py main