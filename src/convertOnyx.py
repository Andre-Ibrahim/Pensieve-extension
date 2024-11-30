import torch.onnx
import ppo2 as network
import sys

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 9
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200, 300, 480, 750, 1200, 1850, 2850, 4300, 5300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DELAY_PENALTY = 10000
BUFFER_THRESH = 8.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000

NN_MODEL = sys.argv[1]

def run():
    net = network.Network(state_dim=[S_INFO, A_DIM], action_dim=A_DIM,
    learning_rate=ACTOR_LR_RATE)

    net.save_in_onnx(NN_MODEL)


run()


