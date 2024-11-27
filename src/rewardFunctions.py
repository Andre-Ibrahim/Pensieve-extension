import numpy as np

VIDEO_BIT_RATE = [235, 375, 560, 750, 1050, 1750, 2350, 3000, 4300]
BUFFER_NORM_FACTOR = 10.0

M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
BUFFER_THRESH = 8.0

class RearwardFunction:
    def reward_with_buffer_and_delay(bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return (VIDEO_BIT_RATE[bit_rate] / 4300 + buffer_size / BUFFER_THRESH) \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K \
                - delay / M_IN_K / BUFFER_NORM_FACTOR
    
    def reward_delay_when_rebuffer(alpha, bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return VIDEO_BIT_RATE[bit_rate] / 4300 \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K \
                - (rebuf > 0) * delay / M_IN_K / BUFFER_NORM_FACTOR

    def reward_with_and_delay(bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return VIDEO_BIT_RATE[bit_rate] / 4300 \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K \
                - delay / M_IN_K / BUFFER_NORM_FACTOR

    def reward_with_buffer_and_delay_alpha(alpha, bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return (alpha * VIDEO_BIT_RATE[bit_rate] / 4300 + (1 - alpha) * buffer_size / BUFFER_THRESH) \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
    
    def reward(alpha, bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return VIDEO_BIT_RATE[bit_rate] / 4300 \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
    
    def reward_with_buffer_no_rebuff(alpha, bit_rate, last_bit_rate, rebuf, delay, buffer_size):
        return (alpha * VIDEO_BIT_RATE[bit_rate] / VIDEO_BIT_RATE[-1] + (1 - alpha) * buffer_size / BUFFER_THRESH) \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K \
                - delay / M_IN_K / BUFFER_NORM_FACTOR
    

    def reward_with_buffer_no_rebuff_switch_rate(alpha, beta, gamma, bit_rate, last_bit_rate, rebuf, delay, buffer_size, switch_rate):
        return (alpha * VIDEO_BIT_RATE[bit_rate] / VIDEO_BIT_RATE[-1] + (1 - alpha) * buffer_size / BUFFER_THRESH) \
                - (beta / (switch_rate + 0.0001)) * SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K \
                - (gamma) * delay / M_IN_K / BUFFER_NORM_FACTOR