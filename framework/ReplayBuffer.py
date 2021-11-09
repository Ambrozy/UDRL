import tensorflow.keras.utils as U
import numpy as np
from sortedcontainers import SortedList


def sample_interval(episode):
    end = len(episode['rewards'])
    start_from = np.random.randint(0, end)
    state = episode['states'][start_from]
    action = episode['actions'][start_from]
    reward = sum(episode['rewards'][start_from:])
    time = end - start_from
    command = [time, reward]

    return state, command, action


def sample_batch(buffer, batch_size):
    indexes = np.random.randint(0, len(buffer), batch_size)
    episodes = [buffer[idx] for idx in indexes]

    state_input = []
    command_input = []
    output = []

    for episode in episodes:
        state, command, action = sample_interval(episode)
        state_input.append(state)
        command_input.append(command)
        output.append(action)

    return [np.array(state_input), np.array(command_input)], np.array(output)


class ReplayBuffer(U.Sequence):
    def __init__(self, max_size, samples_per_epoch, batch_size):
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.max_size = max_size
        self.buffer = SortedList(key=lambda episode: episode['summed_reward'])

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __getitem__(self, index):
        return sample_batch(self.buffer, self.batch_size)

    def add(self, states, actions, rewards):
        self.buffer.add({
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'summed_reward': sum(rewards),
        })

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
