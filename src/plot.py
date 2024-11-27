import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, savefig
import matplotlib
import sys
from collections import OrderedDict
import scipy.stats
plt.switch_backend('agg')

NUM_BINS = 500
BITS_IN_BYTE = 8.0
MILLISEC_IN_SEC = 1000.0
M_IN_B = 1000000.0
VIDEO_LEN = 500
K_IN_M = 1000.0
REBUF_P = 4.3
SMOOTH_P = 1

# labels = SCHEMES#, 'RB']
LW = 1.5
LOG = './baselines/'

SCHEMES = ['alpha_40_hete_vid', 'retrain_vs_heterogenous', 'heterogenous_switch_rate_beta_25', 'alpha_25_buff_size_no_delay', 'alpha_40_buff_size_no_delay', 'alpha_30_buff_size_no_delay', 'alpha_325_buff_size_no_delay', 'heterogenous_switch_rate', 'retrain_hete_vid']
labels = ['alpha_40_hete_vid', 'ret_vs_het', 'hete_swi_ra_beta_25', 'alpha_25', 'alpha_40', 'alpha_30', 'alpha_325', 'heterogenous_switch_rate', 'retrain_hete_vid']
lines = ['-', '--', '-.', ':', '-', '--', ':', '--', ':', '-']  # '--', '-.', ':', '-', '--'
modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#FF9DA7', '#9C755F', '#000000']  # '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F'

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def inlist(filename, traces):
    ret = False
    for trace in traces:
        if trace in filename:
            ret = True
            break
    return ret

def bitrate_smo(outputs):
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    SCHEMES = ['ppo_retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    labels = ['ppo retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    lines = ['-', '--', '-.', ':', '-', '--']  # '--', '-.', ':', '-', '--'
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Bitrate Smoothness (mbps)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()
    # ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def smo_rebuf(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    SCHEMES = ['ppo_retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    labels = ['ppo retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    lines = ['-', '--', '-.', ':', '-', '--']  # '--', '-.', ':', '-', '--'
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']
    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_smo)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Bitrate Smoothness (mbps)')
    ax.set_ylim(0.05, max_bitrate + 0.05)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()
    ax.invert_yaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def bitrate_rebuf(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    SCHEMES = ['ppo_retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    labels = ['ppo retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    lines = ['-', '--', '-.', ':', '-', '--']  # '--', '-.', ':', '-', '--'
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'] # '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F'

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(6, 4))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        mean_bit = []
        mean_rebuf = []
        mean_smo = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                bitrate, rebuffer = [], []
                time_all = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        rebuffer.append(float(sp[3]))
                        arr.append(float(sp[-1]))
                        time_all.append(float(sp[0]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
                mean_bit.append(np.mean(bitrate[:]))
                mean_rebuf.append(np.sum(rebuffer[1:]) / (VIDEO_LEN * 4. + np.sum(rebuffer[1:])) * 100.)
                mean_smo.append(np.mean(np.abs(np.diff(bitrate))))
        reward_all[scheme] = mean_arr
        mean_, low_, high_ = mean_confidence_interval(mean_bit)
        mean_rebuf_, low_rebuf_, high_rebuf_ = mean_confidence_interval(mean_rebuf)
        # mean_smo_, low_smo_, high_smo_ = mean_confidence_interval(mean_smo)
        
        max_bitrate = max(high_, max_bitrate)
        
        ax.errorbar(mean_rebuf_, mean_, \
            xerr= high_rebuf_ - mean_rebuf_, yerr=high_ - mean_, \
            color = modern_academic_colors[idx],
            marker = markers[idx], markersize = 10, label = labels[idx],
            capsize=4)

        out_str = '%s %.2f %.2f %.2f %.2f %.2f %.2f'%(scheme, mean_, low_, high_, mean_rebuf_, low_rebuf_, high_rebuf_)
        print(out_str)

    ax.set_xlabel('Time Spent on Stall (%)')
    ax.set_ylabel('Video Bitrate (mbps)')
    ax.set_ylim(max_bitrate * 0.5, max_bitrate * 1.01)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower left')
    ax.invert_xaxis()

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def qoe_cdf(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    SCHEMES = ['ppo_retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    labels = ['ppo retrained', 'ppo_buffer_size_delay', 'ppo_delay_when_rebuffer', 'ppo_tuned_delay', 'ppo_reward_buffsize_no_rebuff']
    lines = ['-', '--', '-.', ':', '-', '--']  # '--', '-.', ':', '-', '--'
    modern_academic_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F'] # '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F'

    reward_all = {}

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    plt.subplots_adjust(left=0.06, bottom=0.16, right=0.96, top=0.96)

    max_bitrate = 0
    for idx, scheme in enumerate(SCHEMES):
        mean_arr = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')
                arr = []
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        arr.append(float(sp[-1]))
                f.close()
                # if np.mean(arr[1:]) > -100.:
                mean_arr.append(np.mean(arr[1:]))
        reward_all[scheme] = mean_arr

        values, base = np.histogram(reward_all[scheme], bins=NUM_BINS)
        cumulative = np.cumsum(values)
        cumulative = cumulative / np.max(cumulative)
        ax.plot(base[:-1], cumulative, '-', \
                color=modern_academic_colors[idx], lw=LW, \
                label='%s: %.2f' % (labels[idx], np.mean(mean_arr)))

        print('%s, %.2f' % (scheme, np.mean(mean_arr)))
    ax.set_xlabel('QoE')
    ax.set_ylabel('CDF')
    ax.set_ylim(0., 1.01)
    ax.set_xlim(0., 1.8)

    ax.grid(linestyle='--', linewidth=1., alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white',loc='lower right')

    fig.savefig(outputs + '.png')
    # fig.savefig(outputs + '.pdf')
    plt.close()

def rebuffering_vs_time(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']

    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_ylim(0, 2)

    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    for idx, scheme in enumerate(SCHEMES):
        time_all = []
        rebuffer_all = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scheme = LOG + '/' + files
                with open(file_scheme, 'r') as f:
                    time = []
                    rebuffer = []
                    t1 = float(f.readline().split()[0])
                    for line in f:
                        sp = line.split()
                        if len(sp) > 1:
                            time.append(float(sp[0]) - t1)
                            rebuffer.append(float(sp[3]))
                time_all.append(time)
                rebuffer_all.append(rebuffer)

        

        # Find the maximum time length to align all rebuffering times
        max_time_length = max(len(t) for t in time_all)

        print(max_time_length)

        # Initialize arrays to store summed rebuffering times and counts
        summed_rebuffer = np.zeros(max_time_length)
        counts = np.zeros(max_time_length)

        # Sum rebuffering times and counts for each time point
        for time, rebuffer in zip(time_all, rebuffer_all):
            for i, t in enumerate(time):
                summed_rebuffer[i] += rebuffer[i]
                counts[i] += 1

        # Calculate mean rebuffering times
        mean_rebuffer = summed_rebuffer / counts

        # Plot rebuffering vs. time
        ax.plot(range(max_time_length), mean_rebuffer, label=labels[idx], linestyle=lines[idx], color=modern_academic_colors[idx])

    ax.set_xlabel('# chunk delivery', color='#000000')
    ax.set_ylabel('average rebuffering', color='#000000')

    ax.grid(linestyle='--', linewidth=1., alpha=0.5, color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white', loc='upper right')

    fig.savefig(outputs + '_rebuff_per_chunk.png')
    plt.close()

def average_quality_per_second(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    
    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    ax.set_ylim(0, 6)

    for idx, scheme in enumerate(SCHEMES):
        time_all = []
        quality_all = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scheme = LOG + '/' + files
                with open(file_scheme, 'r') as f:
                    time = []
                    quality = []
                    t1 = float(f.readline().split()[0])
                    for line in f:
                        sp = line.split()
                        if len(sp) > 1:
                            time.append(float(sp[0]) - t1)
                            quality.append(float(sp[1]) / 1000.0) 
                time_all.append(time)
                quality_all.append(quality)

        # Find the maximum time length to align all quality times
        max_time_length = max(len(t) for t in time_all)

        # Initialize arrays to store summed quality and counts
        summed_quality = np.zeros(max_time_length)
        counts = np.zeros(max_time_length)

        # Sum quality and counts for each time point
        for time, quality in zip(time_all, quality_all):
            for i, t in enumerate(time):
                summed_quality[i] += quality[i]
                counts[i] += 1

        # Calculate mean quality
        mean_quality = summed_quality / counts

        # Plot average quality per second
        ax.plot(range(max_time_length), mean_quality, label=labels[idx], linestyle=lines[idx], color=modern_academic_colors[idx])

    ax.set_xlabel('# chunk delivery', color='#000000')
    ax.set_ylabel('Average Quality (Mbps)', color='#000000')

    ax.grid(linestyle='--', linewidth=1., alpha=0.5, color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white', loc='upper right')

    fig.savefig(outputs + 'q_per_chunk.png')
    plt.close()

def average_smothness_per_second(outputs):
    # os.system('cp ./test_results/* ' + LOG)
    markers = ['o','x','v','^','>','<','s','p','*','h','H','D','d','1']
    plt.rcParams['axes.labelsize'] = 15
    font = {'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.subplots_adjust(left=0.14, bottom=0.16, right=0.96, top=0.96)

    for idx, scheme in enumerate(SCHEMES):
        time_all = []
        smo_all = []
        for files in os.listdir(LOG):
            if scheme in files:
                file_scehem = LOG + '/' + files
                f = open(file_scehem, 'r')

                bitrate, time, smo = [], [], []
                t1 = float(f.readline().split()[0])
                for line in f:
                    sp = line.split()
                    if len(sp) > 1:
                        bitrate.append(float(sp[1]) / 1000.)
                        t = float(sp[0]) - t1
                        time.append(t)
                        s = np.mean(np.abs(np.diff(bitrate)))
                        if np.isnan(s):
                            s = 0
                        smo.append(s)
                time_all.append(time)
                smo_all.append(smo)
                f.close()

        max_time_length = max(len(t) for t in time_all)

        # Initialize arrays to store summed quality and counts
        summed_smo = np.zeros(max_time_length)
        counts = np.zeros(max_time_length)

        # Sum quality and counts for each time point
        for time, smo in zip(time_all, smo_all):
            for i, t in enumerate(time):
                summed_smo[i] += smo[i]
                counts[i] += 1

        # Calculate mean quality
        mean_quality = summed_smo / counts

        # Plot average quality per second
        ax.plot(range(max_time_length), summed_smo, label=labels[idx], linestyle=lines[idx], color=modern_academic_colors[idx])

    ax.set_xlabel('# chunk delivery', color='#000000')
    ax.set_ylabel('smotheness', color='#000000')

    ax.grid(linestyle='--', linewidth=1., alpha=0.5, color='#000000')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, ncol=3, edgecolor='white', loc='upper right')

    fig.savefig(outputs + '_smo_per_chunk.png')
    plt.close()

if __name__ == '__main__':
    os.system('cp ./test_results/* ' + LOG)
    # bitrate_rebuf('baselines-br')
    # smo_rebuf('baselines-sr')
    # bitrate_smo('baselines-bs')
    # qoe_cdf('baselines-qoe')
    rebuffering_vs_time('homogenous')
    average_quality_per_second('homogenous')
    average_smothness_per_second('homogenous')