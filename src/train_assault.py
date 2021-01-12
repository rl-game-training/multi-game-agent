import gym
from queue import Queue
from random import randint
import random
import numpy as np
import torchvision.transforms as T
import torch
from copy import deepcopy
from network import DQN
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state'))

def predict_action(obs, dqn_policy):
    
    dqn_policy.eval()
    frames = torch.cat(list(obs)).float().clamp(0,1)
    frames = frames.view(1, *frames.size())
    action = dqn_policy(frames).max(1)[1].detach()
    dqn_policy.train()
    return action

class Buffer:

    def __init__(self, capacity):

        self.storage = deque()
        self.write_ptr = 0
        self.capacity = capacity
        self.size = 0

    def insert(self, entry):
        
        self.storage.append(entry)
        if len(self.storage) > self.capacity:
            self.storage.popleft()
            
        
        #print("write to ", self.write_ptr % self.capacity, self.capacity)
        self.write_ptr += 1

    def sample_random(self, batch_size):

        pass
        

transform_frame = T.Compose([
                T.ToTensor(),
                T.Resize(80),
                T.Grayscale()])
            
def preprocess_frame(frame):

    transformed = (transform_frame(frame)*255).to(device)
    return transformed.byte()


GAMMA = 0.99

def update_net(dqn_policy, dqn_target, optimizer):

    def get_states_tensor(transitions, next=False):

        idx = 0
        if next: idx = 3

        return torch.stack(list(map(lambda x: torch.cat(list(x[idx].storage)), transitions))).float().to(device)

    transitions_batch = random.sample(replay_buffer.storage, TRANSITIONS_BATCH_SIZE)

    #create a mask to more easily pick final and non-final states from torch tensors
    non_final_transitions_mask = torch.tensor([idx for idx, trans in enumerate(transitions_batch) if trans.next_state != None]).to(device)

    non_final_transitions = [trans for trans in transitions_batch if trans.next_state != None]


    states = get_states_tensor(transitions_batch).clamp(0, 1)
    next_states = get_states_tensor(non_final_transitions, next=True).clamp(0, 1)
    
    #print("states size", states.size())
    #print("next states size ", next_states.size())
    #print("states equal", torch.equal(states, next_states))
    #print("nest states", next_states.sum())

    actions = torch.tensor(list(map(lambda x: x.action, transitions_batch)), device=device)


    y = torch.tensor(list(map(lambda x: x.reward, transitions_batch)), device=device)
    #print("y rewards range", y.sum())
    q_prediction = dqn_target(next_states).max(1)[0].detach()
    #print("q prediction", q_prediction.sum())
    y[non_final_transitions_mask] += GAMMA * q_prediction

    dqn_policy.eval()
    policy_pred = dqn_policy(states).gather(1, actions.view(-1, 1)).view(y.size())
    dqn_policy.train()
    policy_pred_target = dqn_target(states).gather(1, actions.view(-1, 1)).view(y.size())

    #print("policy pred ", policy_pred.sum())
    #print("policy pred target", policy_pred_target.sum())
    #print("y - x", (policy_pred-y).sum())
    #input()
    huber_loss = F.smooth_l1_loss(policy_pred, y, reduction='mean')

    optimizer.zero_grad()
    huber_loss.backward()
    for param in dqn_policy.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    #print(huber_loss)
    return huber_loss


env = gym.make("Breakout-v0")

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
print(env.unwrapped.get_action_meanings())


REPLAY_BUFFER_LEN = 20000
TRANSITIONS_BATCH_SIZE = 128

NET_W, NET_H, OUTPUT_LEN = (105, 80, 4)
SYNC_TARGET_FREQ = 10

dqn_policy = DQN(NET_W, NET_H, OUTPUT_LEN).to(device)
dqn_target = DQN(NET_W, NET_H, OUTPUT_LEN).to(device)
dqn_target.load_state_dict(dqn_policy.state_dict())
dqn_target.eval()

optimizer = optim.RMSprop(dqn_policy.parameters())
replay_buffer = Buffer(capacity=REPLAY_BUFFER_LEN)
frame_buffer = Buffer(capacity=4)

#file to save reward history
reward_history = "reward_history"
best_reward = 0

def buffers_diff(buf1, buf2):

    buf1_tens = torch.cat(list(buf1.storage)).long()
    buf2_tens = torch.cat(list(buf2.storage)).long()
    print("buf1 sum, size ", buf1_tens.sum(), buf1_tens.size())
    print("buf2 sum, size", buf2_tens.sum(), buf2_tens.size())
    
    print("diff sum", (buf1_tens - buf2_tens).sum())

def iterate_train(num_episodes):

    for ep in range(num_episodes):
        
        entry_frame = preprocess_frame(env.reset())
        reward_sum = 0
        frames = 0
        loss = 0
        while True:

            #print(loss)
            env.render()
            
            frame_buffer.insert(entry_frame)

            # 1% chance to make random action
            if randint(1, 100) == 1 or len(frame_buffer.storage) < frame_buffer.capacity:
                action = env.action_space.sample()

            else:
                action = predict_action(frame_buffer.storage, dqn_policy)
            
            next_frame, reward, done, info = env.step(action)
            next_frame_buffer = None

            if not done:
                next_frame = preprocess_frame(next_frame)
                #print("next frame sum ", next_frame.sum())
                #print("curr frame sum", entry_frame.sum())
                #print("entry next diff", (entry_frame.long() - next_frame.long()).sum())
                next_frame_buffer = deepcopy(frame_buffer)
                next_frame_buffer.insert(next_frame)
                entry_frame = deepcopy(next_frame)

            #put transition to experience replay buffer
            if len(frame_buffer.storage) >= frame_buffer.capacity:
                transition = Transition(frame_buffer, action, reward, next_frame_buffer)
                replay_buffer.insert(transition)

            #if len(frame_buffer.storage) == frame_buffer.capacity: buffers_diff(frame_buffer, next_frame_buffer)
            
            #sample random transitions calculate loss and update weights
            if len(replay_buffer.storage) >= TRANSITIONS_BATCH_SIZE:
                loss += update_net(dqn_policy, dqn_target, optimizer)
            
            reward_sum += reward
            if done:
                print("episode done, reward: ", reward_sum)
                if frames > 0: print("loss: ", loss/frames)
                frames = 0
                loss = 0
                with open(reward_history, "a") as f:
                    f.write(str(reward_sum) + '\n')

                if reward_sum > best_reward:
                    torch.save(dqn_policy, "best_net")

                break
            
            if frames % SYNC_TARGET_FREQ == 0:
                
                dqn_target.load_state_dict(dqn_policy.state_dict()) 

            frames += 1

if __name__ == '__main__':
    
    iterate_train(10000)



    env.close()
