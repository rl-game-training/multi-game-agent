import copy
import os
import random
import re

import numpy as np
import torch
import torch.nn as nn

from collections import deque, namedtuple
from skimage.color import rgb2gray
from skimage.transform import rescale
from tqdm import tqdm_notebook

DEFAULT_CHECKPOINT_DIR = 'ckpt'

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'next_state'])


class AgentMemory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def insert(self, *args):
        self.memory.append(Transition(*args))

    def get_batch(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, done, next_state = map(torch.cat, [*batch])
        return state, action, reward, done, next_state


class BreakoutAgent:
    def __init__(self,
                 env,
                 model,
                 memory_size,
                 batch_size,
                 num_frames,
                 gamma,
                 learning_rate,
                 burn_in_steps,
                 explore_start,
                 explore_stop,
                 decay_rate,
                 update_interval,
                 save_interval,
                 clone_interval,
                 ckpt_dir=DEFAULT_CHECKPOINT_DIR
                 ):
        # CUDA device available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        # ==== Constructor params ====

        # Agent environment
        self.env = env
        # DQN model (enable cuda if available)
        self.model = model.cuda() if self.cuda else model
        # Agent memory
        self.memory = AgentMemory(memory_size)
        self.batch_size = batch_size

        # Environment details
        self.num_actions = self.env.action_space.n
        self.num_frames = num_frames

        # Learning params
        self.gamma = torch.tensor([gamma], device=self.device)
        self.learning_rate = learning_rate
        self.burn_in_steps = burn_in_steps  # How many steps of random actions before starting to train
        # Exploration params
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate

        # Directory where training checkpoints are saved
        self.ckpt_dir = ckpt_dir
        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)

        # Agent details
        self.update_interval = update_interval  # Interval at which the DQN is updated
        self.save_interval = save_interval  # Interval at which checkpoints are saved
        self.clone_interval = clone_interval  # Interval at which model is cloned
        # Steps the agent has done
        self.total_steps = 0

        self.loss = nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                             lr=learning_rate,
                                             alpha=0.95,
                                             eps=0.01)

        self.clone_model()

    def load_ckpt(self):
        """
        Loads the latest checkpoint from the specified checkpoint directory, if there are any.
        """
        ckpt_files = [file for file in os.listdir(self.ckpt_dir) if 'model' in file]
        if len(ckpt_files) < 1:
            print('No checkpoint files found.')
        else:
            recorded_steps = [int(re.search('\d+', file).group(0)) for file in ckpt_files]
            # Get the file with the largest steps
            latest_ckpt = ckpt_files[np.argmax(recorded_steps)]
            # Load steps from checkpoint
            self.total_steps = np.max(recorded_steps)

            print('Loading checkpoint: {}'.format(latest_ckpt))
            self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, latest_ckpt)))

    def save_ckpt(self):
        """
        Saves a checkpoint in the specified checkpoint directory.
        """
        save_path = os.path.join(self.ckpt_dir, 'model-{}'.format(self.total_steps))
        torch.save(self.model.state_dict(), save_path)

    def process_state(self, state):
        """
        Pre-processing of the environment state.
        Converts to grayscale and rescales it to 0.5 of the original frame.

        :param state: State to process
        :return: The processed state
        """

        # Crop the top and bottom parts of our frame since they contain no valuable information.
        state = state[35:195, :, :]
        # Greyscale the frame to reduce the memory print 3 times
        state = rgb2gray(state)
        # Rescale
        state = rescale(state, scale=0.5)
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def predict_action(self, state, train=False):
        """
        Predict an action to take with an exploration probability using epsilon greedy
        strategy

        :param state: Current state.
        :param train: Flag if the agent is training. If set to False exploration is ignored.
        :return: Predicted action
        """
        decay_steps = self.total_steps - self.burn_in_steps
        tradeoff = np.random.rand()
        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * \
                              np.exp(-self.decay_rate * decay_steps)

        if train and explore_probability > tradeoff:
            return np.random.choice(self.num_actions)
        else:
            q_values = self.model(state)
            action = torch.argmax(q_values)
            return action.item()

    def clone_model(self):
        try:
            del self.model_cloned
        except:
            pass

        self.model_cloned = copy.deepcopy(self.model)

        for param in self.model_cloned.parameters():
            param.requires_grad = False

        if self.cuda:
            self.model_cloned = self.model_cloned.cuda()

    def update_dqn(self):
        """
        Updates the weights of the DQN model.
        """
        self.model.zero_grad()

        state, action, reward, done, next_state = self.memory.get_batch(self.batch_size)
        q = self.model(state).gather(1, action.view(self.batch_size, 1))
        q_max = self.model_cloned(next_state).max(dim=1)[0]

        target = done.float() * reward + (~done).float() * (reward + self.gamma * q_max)

        loss = self.loss(q.view(-1), target)
        loss.backward()
        self.optimizer.step()

    def train(self, episodes, load_ckpt=False, render=False):
        """
        Starts the agent's training process.

        :param episodes: Episodes for the agent to train
        :param load_ckpt: If the agent should look for a saved checkpoint
        :param render: If the agent should render the environment
        """
        # Reset the steps
        self.total_steps = 0
        metadata = dict(episode=[], reward=[])

        if load_ckpt:
            self.load_ckpt()

        try:
            progress_bar = tqdm_notebook(range(episodes), unit='episode')
            steps = 0

            for episode in progress_bar:

                state = self.env.reset()
                state = self.process_state(state)

                is_new_episode = True
                done = False
                total_reward = 0

                while not done:
                    if render:
                        self.env.render()

                    # Our state consists of 4 stacked game frames. If we are in a new episode,
                    # stack the current frame 4 times and proceed from there.
                    if is_new_episode:
                        state_copy = copy.deepcopy(state)
                        while state.size()[1] < self.num_frames:
                            state = torch.cat([state, state_copy], 1)
                        is_new_episode = False

                    action = self.predict_action(state, train=True)

                    # Execute predicted action
                    new_state, reward, done, _ = self.env.step(action)
                    # Pre-process new state
                    new_state = self.process_state(new_state)

                    # Append the new frame (state)
                    new_state = torch.cat([state, new_state], 1)
                    # Remove the oldest frame
                    new_state = new_state[:, 1:, :, :]

                    # Convert to tensors
                    reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                    action = torch.tensor([action], device=self.device, dtype=torch.long)
                    done = torch.tensor([done], device=self.device, dtype=torch.uint8)

                    # Remember the transition
                    self.memory.insert(state, action, reward, done, new_state)

                    state = new_state
                    total_reward += reward
                    self.total_steps += 1
                    steps += 1

                    if self.total_steps > self.burn_in_steps and steps > self.batch_size:
                        if self.total_steps % self.update_interval == 0:
                            self.update_dqn()

                        if self.total_steps % self.clone_interval == 0:
                            self.clone_model()

                        if self.total_steps % self.save_interval == 0:
                            self.save_ckpt()

                    # Update progress bar every 1000 steps
                    if self.total_steps % 1000 == 0:
                        progress_bar.set_description('total_steps={}'.format(self.total_steps))

                metadata['episode'].append(episode)
                metadata['reward'].append(total_reward)

                # Log info every 100 episodes
                if episode % 100 == 0 and episode != 0:
                    avg_reward = np.mean(metadata['reward'][-100:], dtype=np.float)
                    print('Average reward (last 100 episodes): {:.2f}'.format(avg_reward))

            self.env.close()
            return metadata
        except KeyboardInterrupt:
            print('Saving model before terminating...')
            self.save_ckpt()
            self.env.close()
            return metadata

    def play(self, episodes, load_ckpt=False, render=False):
        """
        Let the agent play using the DQN to predict actions.

        :param episodes: Episodes (number of games) the agent will play.
        :param load_ckpt: If the agent should load a checkpoint.
        :param render: If the agent should render the environment.
        :return:
        """
        # Reset the steps
        self.total_steps = 0
        metadata = dict(episode=[], reward=[])

        if load_ckpt:
            self.load_ckpt()

        try:
            progress_bar = tqdm_notebook(range(episodes), unit='episode')

            for episode in progress_bar:

                state = self.env.reset()
                state = self.process_state(state)

                done = False
                is_new_episode = True
                episode_steps = 0
                total_reward = 0

                while not done:
                    if render:
                        self.env.render()

                    # Our state consists of 4 stacked game frames. If we are in a new episode,
                    # stack the current frame 4 times and proceed from there.
                    if is_new_episode:
                        state_copy = copy.deepcopy(state)
                        while state.size()[1] < self.num_frames:
                            state = torch.cat([state, state_copy], 1)
                        is_new_episode = False

                    action = self.predict_action(state)
                    new_state, reward, done, _ = self.env.step(action)
                    new_state = self.process_state(new_state)

                    new_state = torch.cat([state, new_state], 1)
                    new_state = new_state[:, 1:, :, :]

                    state = new_state
                    total_reward += reward

                    print('Completed step {}. Action taken: {}. Reward gained: {}'
                          .format(self.total_steps, action, reward))

                    self.total_steps += 1
                    episode_steps += 1

                metadata['episode'].append(episode)
                metadata['reward'].append(total_reward)

                print('Play episode completed in {} steps. Reward: {:.2f}'.format(episode_steps,
                                                                                  total_reward))

            self.env.close()
            return metadata
        except KeyboardInterrupt:
            self.env.close()
            return metadata
