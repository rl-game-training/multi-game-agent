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
from collections import namedtuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state'))

def predict_action(obs, dqn_policy):
    
    return dqn_policy(dqn_input_states2).max(1)[0].detach()

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

    transformed = transform_frame(frame).to(device)
    return transformed.byte()


GAMMA = 0.99

def update_net(dqn_policy, dqn_target, optimizer):

    def get_states_tensor(transitions, next=False):

        idx = 0
        if next: idx = 3

        return torch.stack(list(map(lambda x: torch.cat(x[idx].storage), transitions))).float().to(device)

    transitions_batch = random.sample(replay_buffer.storage, TRANSITIONS_BATCH_SIZE)

    #create a mask to more easily pick final and non-final states from torch tensors
    final_transitions_mask = torch.tensor([idx for idx, trans in enumerate(transitions_batch) if trans.next_state == None]).to(device)
    non_final_transitions_mask = torch.tensor([idx for idx, trans in enumerate(transitions_batch) if trans.next_state != None]).to(device)

    final_transitions = [trans for trans in transitions_batch if trans.next_state == None]
    non_final_transitions = [trans for trans in transitions_batch if trans.next_state != None]


    states = get_states_tensor(transitions_batch)
    #transform list<tuple<Buffer.storage>> into torch tensor
    next_states = get_states_tensor(non_final_transitions, next=True)
    next_states /= 256

    actions = torch.tensor(list(map(lambda x: x.action, transitions_batch)), device=device)
    #y[final_mask] = rewards[final_mask]
    #y[non_final_mask] = rewards[non_final_mask] + GAMMA * dqn_target(next_states)[non_final_mask]

    y = torch.tensor(list(map(lambda x: x.reward, transitions_batch)), device=device)
    #print("y before pred ", y)
    y[non_final_transitions_mask] += GAMMA * dqn_target(next_states).max(1)[0].detach()
    #print("y ", y)
    #print("non final transitions", non_final_transitions_mask)

    #print("actions", actions)

    policy_pred = dqn_policy(states).gather(1, actions.view(-1, 1))

    #print("policy pred size", policy_pred.size())

    huber_loss = F.smooth_l1_loss(policy_pred, y, reduction='mean')
    #print("loss", huber_loss)
     # Optimize the model
    optimizer.zero_grad()
    huber_loss.backward()
    for param in dqn_policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return huber_loss


env = gym.make("Breakout-v0")

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
#print(env.unwrapped.get_action_meanings())


REPLAY_BUFFER_LEN = 6000
TRANSITIONS_BATCH_SIZE = 30

NET_W, NET_H, OUTPUT_LEN = (105, 80, 7)
SYNC_TARGET_FREQ = 10

dqn_policy = DQN(NET_W, NET_H, OUTPUT_LEN).to(device)
dqn_target = DQN(NET_W, NET_H, OUTPUT_LEN).to(device)
dqn_target.eval()

optimizer = optim.RMSprop(dqn_policy.parameters())
replay_buffer = Buffer(capacity=REPLAY_BUFFER_LEN)
frame_buffer = Buffer(capacity=4)

#file to save reward history
reward_history = "reward_history"
best_reward = 0


def iterate_train(num_episodes):

    for ep in range(num_episodes):
        
        entry_frame = preprocess_frame(env.reset())
        reward_sum = 0
        frames = 0
        loss = 0
        while True:

            env.render()
            
            frame_buffer.insert(entry_frame)

            # 1% chance to make random action
            if randint(1, 100) == 1 or frame_buffer.size < frame_buffer.capacity:
                action = env.action_space.sample()

            else:
                action = predict_action(frame_buffer.storage, dqn_policy)
            
            next_frame, reward, done, info = env.step(action)
            next_frame_buffer = None

            if not done:
                next_frame = preprocess_frame(next_frame)
                next_frame_buffer = copy(frame_buffer)
                next_frame_buffer.insert(next_frame)
                entry_frame = copy(next_frame)

            #put transition to experience replay buffer
            transition = Transition(frame_buffer, action, reward, next_frame_buffer)

            replay_buffer.insert(transition)
                    #sample random transitions calculate loss and update weights
            if replay_buffer.size >= 100:
                loss += update_net(dqn_policy, dqn_target, optimizer)
            
            if frames % 50 == 0:
                
                print(loss/50)
                loss = 0
            
            reward_sum += reward
            if done:
                print("episode done, reward: ", reward_sum)
                with open(reward_history, "a") as f:
                    f.write(str(reward_sum) + '\n')

                if reward_sum > best_reward:
                    torch.save(dqn_policy, "best_net")

                print(len(replay_buffer.storage))
                break
            
            if frames % SYNC_TARGET_FREQ == 0:
                
                dqn_target.load_state_dict(dqn_policy.state_dict()) 

            frames += 1

if __name__ == '__main__':
    
    iterate_train(2)



    env.close()
