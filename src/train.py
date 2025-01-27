import multiprocessing as mp
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from env import ABREnv
import ppo2 as network
import torch
from tqdm import trange

from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler

S_DIM = [6, 9]
A_DIM = 9
ACTOR_LR_RATE = 1e-4
NUM_AGENTS = 1
TRAIN_SEQ_LEN = 1000  # take as a train batch
TRAIN_EPOCH = 5000
MODEL_SAVE_INTERVAL = 100
RANDOM_SEED = 42
NAME = 'hetereogenous_reward5_experiment4_rebuf2'
SUMMARY_DIR = f'./ppo/{NAME}'
MODEL_DIR = './models'
TRAIN_TRACES = './train_hetereogenous/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'
mp.set_start_method('spawn', force=True)

# Hyperparameters to the reward function
ALPHA = 0.8170262751188082
BETA = 0.20
GAMMA = 0.27566949441816124

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

NN_MODEL = None    

def testing(epoch, nn_model, log_file, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    # clean up the test results folder
    #os.system('rm -r ' + TEST_LOG_FOLDER + f"a_{epoch}/")
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system(f'python test.py {nn_model} {str(alpha)} {str(beta)} {str(gamma)} {"True"} {NAME}')

    # append test performance to the log
    rewards, entropies = [], []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards.append(np.mean(reward[1:]))
        entropies.append(np.mean(entropy[1:]))

    rewards = np.array(rewards)

    rewards_min = 0
    rewards_5per = 0
    rewards_mean = 0
    rewards_median = 0
    rewards_95per = 0
    rewards_max = 0

    if len(rewards) != 0:
        rewards_min = np.min(rewards)
        rewards_5per = np.percentile(rewards, 5)
        rewards_mean = np.mean(rewards)
        rewards_median = np.percentile(rewards, 50)
        rewards_95per = np.percentile(rewards, 95)
        rewards_max = np.max(rewards)

    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()

    return rewards_mean, np.mean(entropies)
        
def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS
    
    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        actor = network.Network(state_dim=S_DIM, 
                                action_dim=A_DIM,
                                learning_rate=ACTOR_LR_RATE)

        writer = SummaryWriter(SUMMARY_DIR)

        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            actor.load_model(nn_model)
            print('Model restored.')
        
        # while True:  # assemble experiences from agents, compute the gradients
        for epoch in trange(TRAIN_EPOCH, desc="Training Epochs"):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)

            s, a, p, r = [], [], [], []
            for i in range(NUM_AGENTS):
                s_, a_, p_, r_ = exp_queues[i].get()
                s += s_
                a += a_
                p += p_
                r += r_
            s_batch = np.stack(s, axis=0)
            a_batch = np.vstack(a)
            p_batch = np.vstack(p)
            v_batch = np.vstack(r)

            actor.train(s_batch, a_batch, p_batch, v_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
                
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth', 
                    test_log_file, ALPHA)

                writer.add_scalar('Entropy Weight', actor._entropy_weight, epoch)
                writer.add_scalar('Reward', avg_reward, epoch)
                writer.add_scalar('Entropy', avg_entropy, epoch)
                writer.flush()


def agent(agent_id, net_params_queue, exp_queue):
    env = ABREnv(agent_id)
    
    env.set_alpha(ALPHA)
    env.set_beta(BETA)
    env.set_gamma(GAMMA)

    actor = network.Network(state_dim=S_DIM, action_dim=A_DIM,
                            learning_rate=ACTOR_LR_RATE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    for epoch in range(TRAIN_EPOCH):
        obs = env.reset()
        s_batch, a_batch, p_batch, r_batch = [], [], [], []
        for step in range(TRAIN_SEQ_LEN):
            s_batch.append(obs)

            action_prob = actor.predict(
                np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

            # gumbel noise
            noise = np.random.gumbel(size=len(action_prob))
            bit_rate = np.argmax(np.log(action_prob) + noise)

            obs, rew, done, info = env.step(bit_rate)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)
            r_batch.append(rew)
            p_batch.append(action_prob)
            if done:
                break
        v_batch = actor.compute_v(s_batch, a_batch, r_batch, done)
        exp_queue.put([s_batch, a_batch, p_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():

    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. PyTorch can use the GPU.")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        
        # Print details of each GPU
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
    else:
        print("CUDA is not available. PyTorch will use the CPU.")

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()
    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
