# utils/replay_buffer.py
import random

class ReplayBuffer:
    """
    一个简单的循环经验回放缓冲区
    """

    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, transition):
        """
        添加一条 transition：
        transition = (obs, act, rew, next_obs, done)
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """
        随机采样一批 transition，返回列表
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
