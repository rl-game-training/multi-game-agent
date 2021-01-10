import gym
from queue import Queue
from random import randint
import random
import numpy as np
import torchvision.transforms as T
import torch
from copy import copy
from network import DQN
import torch.nn.functional as F
import torch.optim as optim


def predict_action(obs):
    
    return randint(0, 6)

class Buffer:

    def __init__(self, capacity):

        self.storage = list()
        self.write_ptr = 0
        self.capacity = capacity
        self.size = 0

    def insert(self, entry):
                    
        if len(self.storage) < self.capacity:
            self.storage.append(entry)
            self.write_ptr += 1
            self.size += 1

            return
            
        self.storage[self.write_ptr % self.capacity] = entry
        self.write_ptr += 1

    def sample_random(self, batch_size):

        pass
        

transform_frame = T.Compose([
                T.ToTensor(),
                T.Resize(80),
                T.Grayscale()])
            
def preprocess_frame(frame):

    transformed = transform_frame(frame)
    return transformed.byte()


def update_net(dqn, optimizer):

    transitions_batch = random.sample(replay_buffer.storage, TRANSITIONS_BATCH_SIZE)
    final_transitions = [trans for trans in transitions_batch if trans[-1] == None]
    non_final_transitions = [trans for trans in transitions_batch if trans[-1] != None]


    #transform list<tuple<Buffer.storage>> into torch tensor
    dqn_input_states2 = torch.stack(list(map(lambda x: torch.cat(x[-1].storage), non_final_transitions))).float()
    dqn_input_states2 /= dqn_input_states2.max()

    y_final = torch.tensor(list(map(lambda x: x[2], final_transitions)))
    y_non_final = torch.tensor(list(map(lambda x: x[2], non_final_transitions)))
    y_non_final += dqn(dqn_input_states2).max(1)[0].detach()
    
    prediction_inp_final = None
    prediction_actions_final = None
    pred_final = None

    if len(final_transitions) > 0:
        prediction_inp_final = torch.stack(list(map(lambda x: torch.cat(x[0].storage), final_transitions))).float()
        prediction_actions_final = torch.tensor(list(map(lambda x: x[1], final_transitions)))
        pred_final = dqn(prediction_inp_final)

    predicion_inp_non_final = torch.stack(list(map(lambda x: torch.cat(x[0].storage), non_final_transitions))).float()
    prediction_actions_non_final = torch.tensor(list(map(lambda x: x[1], non_final_transitions)))

    predicion_inp_non_final /= predicion_inp_non_final.max()
    pred_non_final = dqn(predicion_inp_non_final)

    pred_non_final = pred_non_final[np.arange(pred_non_final.size(0)),prediction_actions_non_final]

    hubert_loss = F.smooth_l1_loss(pred_non_final, y_non_final, size_average = True)

     # Optimize the model
    optimizer.zero_grad()
    hubert_loss.backward()
    for param in dqn.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    

env = gym.make("Breakout-v0")

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
#print(env.unwrapped.get_action_meanings())
input()

REPLAY_BUFFER_LEN = 3000
TRANSITIONS_BATCH_SIZE = 30
dqn = DQN(105, 80, 7)
optimizer = optim.RMSprop(dqn.parameters())
replay_buffer = Buffer(capacity=REPLAY_BUFFER_LEN)
frame_buffer = Buffer(capacity=4)

#file to save reward history
reward_history = "reward_history"
best_reward = 0

#training loop
for ep in range(200):

    entry_frame = preprocess_frame(env.reset())
    reward_sum = 0
    while True:

        env.render()
        
        frame_buffer.insert(entry_frame)

        # 1% chance to make random action
        if randint(1, 100) == 1 or frame_buffer.size < frame_buffer.capacity:
            action = env.action_space.sample()

        else:
            action = predict_action(frame_buffer.storage)
        
        next_frame, reward, done, info = env.step(action)
        next_frame_buffer = None

        if not done:
            next_frame = preprocess_frame(next_frame)
            next_frame_buffer = copy(frame_buffer)
            next_frame_buffer.insert(next_frame)
            entry_frame = copy(next_frame)

        #put transition to experience replay buffer
        transition = (frame_buffer, action, reward, next_frame_buffer)

        replay_buffer.insert(transition)
                #sample random transitions calculate loss and update weights
        if replay_buffer.size >= 100:
            update_net(dqn, optimizer)
        
        reward_sum += reward
        if done:
            print("episode done, reward: ", reward_sum)
            with open(reward_history, "a") as f:
                f.write(str(reward_sum) + '\n')

            if reward_sum > best_reward:
                torch.save(dqn, "best_net")
            break


env.close()
