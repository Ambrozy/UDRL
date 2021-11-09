import numpy as np

from framework import BehaviorFunction, ReplayBuffer


class Driver:
    def __init__(self, env, behavior_function: BehaviorFunction, replay_buffer: ReplayBuffer):
        self.env = env
        self.behavior_function = behavior_function
        self.replay_buffer = replay_buffer

    def get_command(self, scale=1, min_reward=1, min_time=1, n_best=20):
        buffer = self.replay_buffer.buffer[-n_best:]

        if len(buffer) == 0:
            return [min_reward, min_time]

        mean_time = np.mean([len(item['rewards']) for item in buffer])
        rewards = [item['summed_reward'] for item in buffer]
        mean_rewards = np.mean(rewards)
        std_rewards = np.std(rewards)

        return [
            mean_time * scale,
            np.random.uniform(mean_rewards, mean_rewards + std_rewards) * scale
        ]

    def train(self):
        history = self.behavior_function.fit(self.replay_buffer, epochs=1, verbose=0)
        return history.history

    def test_episode(self, command, greedy=False, min_time=0, min_reward=0):
        # Prepare initial values
        state = self.env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        expected_time, expected_reward = command
        done = False
        # Run episode
        while not done:
            episode_states.append(state)
            # Get action
            command = [expected_time, expected_reward]
            if greedy:
                action = self.behavior_function.greedy_action(state, command, training=False)
            else:
                action = self.behavior_function.action(state, command, training=False)
            action = action[0].numpy()
            # Get Observation
            state, reward, done, _ = self.env.step(action)
            # Update command components
            expected_time = max(expected_time - 1, min_time)
            expected_reward = max(expected_reward - reward, min_reward)
            # Apply history
            episode_rewards.append(reward)
            episode_actions.append(action)

        return episode_states, episode_actions, episode_rewards

    def evaluate(self, command):
        _, __, rewards = self.test_episode(command, greedy=True, min_time=1, min_reward=1)
        return sum(rewards)

    def upside_down(self, n_episodes, n_test_games, verbose=0):
        history = []
        for episode in range(1, n_episodes + 1):
            # Run test games and add them to the replay buffer
            for _ in range(n_test_games):
                command = self.get_command()
                states, actions, rewards = self.test_episode(command)
                self.replay_buffer.add(states, actions, rewards)
            # Train on data from replay buffer
            local_history = self.train()
            loss = local_history['loss'][0]
            # Evaluate model
            command = self.get_command()
            reward = self.evaluate(command)
            # Store history
            history.append({
                'loss': loss,
                'rewards': reward,
                'command_time': command[0],
                'command_reward': command[1]
            })
            # Print train state
            if verbose:
                print(
                    f'[{episode}]: loss={loss}, greedy_reward={reward:.2f}, ' +
                    f'command_time={command[0]:.2f}, command_reward={command[1]:.2f}'
                )

        return {
            'loss': [h['loss'] for h in history],
            'rewards': [h['rewards'] for h in history],
            'command_time': [h['command_time'] for h in history],
            'command_reward': [h['command_reward'] for h in history]
        }
