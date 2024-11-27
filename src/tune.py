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
TRAIN_EPOCH = 80001
MODEL_SAVE_INTERVAL = 10000
RANDOM_SEED = 42
NAME = 'heterogenous_switch_rate_beta_25'
SUMMARY_DIR = f'./ppo/{NAME}'
MODEL_DIR = './models'
TRAIN_TRACES = './train_hetereogenous/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'
mp.set_start_method('spawn', force=True)

ALPHA = 0.4
BETA = .25
GAMMA = 1

def testing(epoch, nn_model, log_file, alpha=ALPHA, beta=BETA, gamma=GAMMA):
    # clean up the test results folder
    #os.system('rm -r ' + TEST_LOG_FOLDER + f"a_{epoch}/")
    #os.system('mkdir ' + TEST_LOG_FOLDER)

    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system(f'python test.py {nn_model} {str(alpha)} {str(beta)} {str(gamma)} {str(epoch)} {NAME}')

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

def tune_parameter(config):

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    writer = SummaryWriter(SUMMARY_DIR)
    
    env = ABREnv()
    env.set_alpha(config["alpha"])
    env.set_beta(config["beta"])
    env.set_gamma(config["gamma"])

    actor_critic = network.Network(state_dim=S_DIM, action_dim=A_DIM, learning_rate=ACTOR_LR_RATE)

    np.random.seed(RANDOM_SEED)
    torch.set_num_threads(1)

    total_reward = 0

    with open(LOG_FILE + '_test.txt', 'w') as test_log_file:
        for epoch in trange(config["epoch"]):
            obs = env.reset()
            s_batch, a_batch, p_batch, r_batch = [], [], [], []
            for step in range(TRAIN_SEQ_LEN):
                s_batch.append(obs)

                action_prob = actor_critic.predict(np.reshape(obs, (1, S_DIM[0], S_DIM[1])))

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

            v_batch = actor_critic.compute_v(s_batch, a_batch, r_batch, done)
            actor_critic.train(np.array(s_batch), np.array(a_batch), np.array(p_batch), np.array(v_batch), epoch)

            total_reward += np.sum(r_batch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor_critic.save_model(SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth')
        
                avg_reward, avg_entropy = testing(epoch,
                    SUMMARY_DIR + '/nn_model_ep_' + str(epoch) + '.pth',
                    test_log_file, 
                    config["alpha"], 
                    config["beta"], 
                    config["gamma"])

                writer.add_scalar('Entropy Weight', actor_critic._entropy_weight, epoch)
                writer.add_scalar('Reward', avg_reward, epoch)
                writer.add_scalar('Entropy', avg_entropy, epoch)
                writer.flush()

    train.report(dict(total_reward=total_reward))

def tune_main():

    search_space = {
    "alpha": tune.uniform(0.0, 1.0),
    "beta": tune.uniform(0.0, 1.0),
    "gamma": tune.uniform(0.0, 1.0),
    "epoch": 10000
    }
        
    analysis = tune.run(
    tune_parameter,
    config=search_space,
    num_samples=10,
    scheduler=ASHAScheduler(metric="total_reward", mode="max"),
    resources_per_trial={"cpu": 8, "gpu": 0} 
    )

    best_config = analysis.get_best_config(metric="total_reward", mode="max")

    with open("tuning_results.txt", "w") as f:
        f.write("Best config:\n")
        f.write(str(best_config) + "\n")
        f.write("\nAll trials:\n")
        for trial in analysis.trials:
            f.write(f"Trial {trial.trial_id}: {trial.config}, Reward: {trial.last_result['total_reward']}\n")

    print("Best config: ", best_config)

if __name__ == '__main__':
    tune_main()
