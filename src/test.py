import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
#import a2c as network
import ppo2 as network
import fixed_env as env
import rewardFunctions as rf


S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 9
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [200, 300, 480, 750, 1200, 1850, 2850, 4300, 5300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 1000
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DELAY_PENALTY = 10000
BUFFER_THRESH = 8.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
TEST_TRACES = './test_heterogenous/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]

ALPHA = float(sys.argv[2])
BETA = float(sys.argv[3])
GAMMA = float(sys.argv[4])

training = sys.argv[5]
name = sys.argv[6]
LOG_FILE = f'./test_results/log_sim_ppo_{name}'
    
def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')


    actor = network.Network(state_dim=[S_INFO, A_DIM], action_dim=A_DIM,
        learning_rate=ACTOR_LR_RATE)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        actor.load_model(NN_MODEL)
        print("Testing model restored.")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, A_DIM))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    entropy_ = 0.5
    video_count = 0

    totalbr = 0
    count = 0

    total_quality = 0

    bitrate_counts = {}

    
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain, switch_rate = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        reward = rf.RearwardFunction.reward5(ALPHA, BETA, GAMMA, bit_rate, last_bit_rate, rebuf, delay, buffer_size, switch_rate)

        r_batch.append(reward)

        last_bit_rate = bit_rate

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(entropy_) + '\t' + 
                        str(reward) + '\n')
        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, A_DIM))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        action_prob = actor.predict(np.reshape(state, (1, S_INFO, A_DIM)))

        noise = np.random.gumbel(size=len(action_prob))
        bit_rate = np.argmax(np.log(action_prob) + noise)

        if(training == 'False'):
            bit_rate = np.argmax(action_prob)
        
        totalbr += np.argmax(action_prob)
        total_quality += VIDEO_BIT_RATE[np.argmax(action_prob)]
        count += 1

        # Assuming `action_prob` is a numpy array with probabilities for each bitrate
        selected_bitrate = np.argmax(action_prob)  # Get the index of the chosen bitrate

        # Update the dictionary
        if selected_bitrate in bitrate_counts:
            bitrate_counts[selected_bitrate] += 1  # Increment count if bitrate already exists
        else:
            bitrate_counts[selected_bitrate] = 1
        
        s_batch.append(state)
        entropy_ = -np.dot(action_prob, np.log(action_prob))
        entropy_record.append(entropy_)

        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, A_DIM)))
            a_batch.append(action_vec)
            # print(np.mean(entropy_record))
            entropy_record = []

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')

    print("average bitrate: ", totalbr/count)
    print("average quality: ", total_quality/count)
    print(count)
    print(bitrate_counts)


if __name__ == '__main__':
    main()
