import gym
from queue import Queue
from random import randint
import random
import numpy as np
import torchvision.transforms as T
import torch
from copy import copy
from network import DQN

def predict_action(obs):
    
    return randint(0, 6)

class Buffer:

    def __init__(self, capacity):

        self.storage = list()
        self.write_ptr = 0
        self.capacity = capacity
        self.size = 0

    def insert(self, entry):
            
        if self.write_ptr < self.capacity:
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


def update_net(dqn):

    transitions_batch = random.sample(replay_buffer.storage, TRANSITIONS_BATCH_SIZE)
    final_transitions = [trans for trans in transitions_batch if trans[-1] == None]
    non_final_transitions = [trans for trans in transitions_batch if trans[-1] != None]

    print("type of non f tranisitions", type(non_final_transitions[0][-1].storage[0]))
    states = [state[-1].storage for state in non_final_transitions]
    print(type(states))
    print(type(states[0]))
    print(type(states[0][-1]))
    hui = [state[-1].storage for state in non_final_transitions]
    print("hui type", type(hui[0]))
    dqn_input_state2 = torch.tensor(np.array([state[-1].storage for state in non_final_transitions]), dtype=torch.uint8)

    y_final = final_transitions[:][2]
    y_non_final = transitions_batch[:][2]
    y_non_final += dqn(dqn_input_state2).max(1)[0].detach()

    

env = gym.make("Assault-v0")

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
#print(env.unwrapped.get_action_meanings())
input()

REPLAY_BUFFER_LEN = 500
TRANSITIONS_BATCH_SIZE = 30
dqn = DQN(80, 80, 7)
replay_buffer = Buffer(capacity=REPLAY_BUFFER_LEN)
frame_buffer = Buffer(capacity=4)

#training loop
for ep in range(200):

    entry_frame = preprocess_frame(env.reset())

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
        print(replay_buffer.size)
        print("type last tuple el", type(transition[-1]))
        #sample random transitions calculate loss and update weights
        if replay_buffer.size >= 100:
            update_net(dqn)
        
        if done:
            print("episode done, reward: ", reward)
            break


env.close()

len