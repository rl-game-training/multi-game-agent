import os
import re
import copy
import random
import numpy as np

import torch
import torch.nn as nn

from IPython import display
from skimage.color import rgb2grey
from skimage.transform import rescale
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from collections import deque, namedtuple

from pyvirtualdisplay import Display


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminal', 'next_state'])


class Agent:
    def __init__(self,
                 model,
                 env,
                 memory_depth,
                 lr,
                 gamma,
                 epsilon_i,
                 epsilon_f,
                 anneal_time,
                 ckptdir,
                 num_frames,
                 burn_in_steps,
                 update_interval,
                 save_interval,
                 clone_interval,
                 batch_size):

        self.cuda = True if torch.cuda.is_available() else False

        self.model = model
        self.env = env
        self.device = torch.device("cuda" if self.cuda else "cpu")

        if self.cuda:
            self.model = self.model.cuda()

        self.memory_depth = memory_depth
        self.gamma = torch.tensor([gamma], device=self.device)
        self.e_i = epsilon_i
        self.e_f = epsilon_f
        self.anneal_time = anneal_time
        self.ckptdir = ckptdir

        self.num_frames = num_frames
        self.num_actions = self.env.action_space.n
        # TODO delete after verification
        print(self.num_actions)
        self.burn_in_steps = burn_in_steps
        self.update_interval = update_interval
        self.save_interval = save_interval
        self.clone_interval = clone_interval
        self.batch_size = batch_size

        if not os.path.isdir(ckptdir):
            os.makedirs(ckptdir)

        self.memory = deque(maxlen=memory_depth)
        self.clone()

        self.loss = nn.SmoothL1Loss()
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=lr, alpha=0.95, eps=0.01)

    def clone(self):
        try:
            del self.clone_model
        except:
            pass

        self.clone_model = copy.deepcopy(self.model)

        for p in self.clone_model.parameters():
            p.requires_grad = False

        if self.cuda:
            self.clone_model = self.clone_model.cuda()

    def remember(self, *args):
        self.memory.append(Transition(*args))

    def retrieve(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        state, action, reward, terminal, next_state = map(torch.cat, [*batch])
        return state, action, reward, terminal, next_state

    @property
    def memories(self):
        return len(self.memory)

    def act(self, state):
        q_values = self.model(state).detach()
        action = torch.argmax(q_values)
        return action.item()

    def process(self, state):
        state = rgb2grey(state[35:195, :, :])
        state = rescale(state, scale=0.5)
        state = state[np.newaxis, np.newaxis, :, :]
        return torch.tensor(state, device=self.device, dtype=torch.float)

    def exploration_rate(self, t):
        if 0 <= t < self.anneal_time:
            return self.e_i - t * (self.e_i - self.e_f) / self.anneal_time
        elif t >= self.anneal_time:
            return self.e_f
        elif t < 0:
            return self.e_i

    def save(self, t):
        save_path = os.path.join(self.ckptdir, 'model-{}'.format(t))
        torch.save(self.model.state_dict(), save_path)

    def load(self):
        ckpts = [file for file in os.listdir(self.ckptdir) if 'model' in file]
        steps = [int(re.search('\d+', file).group(0)) for file in ckpts]

        latest_ckpt = ckpts[np.argmax(steps)]
        self.t = np.max(steps)

        print("Loading checkpoint: {}".format(latest_ckpt))

        self.model.load_state_dict(torch.load(os.path.join(self.ckptdir, latest_ckpt)))

    def update(self, batch_size):
        self.model.zero_grad()

        state, action, reward, terminal, next_state = self.retrieve(batch_size)
        q = self.model(state).gather(1, action.view(batch_size, 1))
        qmax = self.clone_model(next_state).max(dim=1)[0]

        nonterminal_target = reward + self.gamma * qmax
        terminal_target = reward

        target = terminal.float() * terminal_target + (~terminal).float() * nonterminal_target

        loss = self.loss(q.view(-1), target)
        loss.backward()
        self.opt.step()

    def play(self, episodes, train=False, load=False, plot=False, render=False, verbose=False,
             render_colab=False, omit_sleep=False):

        if render_colab:
            v_display = Display(visible=0, size=(640, 480))
            v_display.start()

        self.t = 0
        metadata = dict(episode=[], reward=[])

        if load:
            self.load()

        try:
            progress_bar = tqdm(range(episodes), unit='episode')

            i = 0
            for episode in progress_bar:

                state = self.env.reset()
                state = self.process(state)

                done = False
                total_reward = 0

                while not done:

                    if render:
                        self.env.render()

                    if render_colab and not render:
                        plt.grid(None)
                        display.clear_output(wait=True)
                        display.display(plt.gcf())
                        plt.imshow(self.env.render(mode='rgb_array'))

                    while state.size()[1] < self.num_frames:
                        action = 1  # Fire

                        new_frame, reward, done, info = self.env.step(action)
                        new_frame = self.process(new_frame)

                        state = torch.cat([state, new_frame], 1)

                    if train and np.random.uniform() < self.exploration_rate(self.t - self.burn_in_steps):
                        action = np.random.choice(self.num_actions)

                    else:
                        action = self.act(state)

                    new_frame, reward, done, info = self.env.step(action)
                    new_frame = self.process(new_frame)

                    new_state = torch.cat([state, new_frame], 1)
                    new_state = new_state[:, 1:, :, :]

                    if train:
                        reward = torch.tensor([reward], device=self.device, dtype=torch.float)
                        action = torch.tensor([action], device=self.device, dtype=torch.long)
                        done = torch.tensor([done], device=self.device, dtype=torch.uint8)

                        self.remember(state, action, reward, done, new_state)

                    state = new_state
                    total_reward += reward
                    self.t += 1
                    i += 1

                    if not train and not omit_sleep:
                        import time
                        time.sleep(0.1)

                    if train and self.t > self.burn_in_steps and i > self.batch_size:

                        if self.t % self.update_interval == 0:
                            self.update(self.batch_size)

                        if self.t % self.clone_interval == 0:
                            self.clone()

                        if self.t % self.save_interval == 0:
                            self.save(self.t)

                    if self.t % 1000 == 0:
                        progress_bar.set_description("t = {}".format(self.t))

                metadata['episode'].append(episode)
                metadata['reward'].append(total_reward)

                if episode % 100 == 0 and episode != 0:
                    avg_return = np.mean(metadata['reward'][-100:], dtype=np.float)
                    print("Average return (last 100 episodes): {:.2f}".format(avg_return))

                if plot:
                    plt.scatter(metadata['episode'], metadata['reward'])
                    plt.xlim(0, episodes)
                    plt.xlabel("Episode")
                    plt.ylabel("Return")
                    display.clear_output(wait=True)
                    display.display(plt.gcf())

            self.env.close()
            return metadata

        except KeyboardInterrupt:
            if train:
                print("Saving model before quitting...")
                self.save(self.t)

            self.env.close()
            return metadata
