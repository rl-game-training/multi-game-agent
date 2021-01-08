import gym
from queue import Queue
from random import randint

def predict_action(obs):
    
    return randint(0, 6)

env = gym.make("CartPole-v1")

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())
#print(env.unwrapped.get_action_meanings())
input()

REPLAY_BUFFER_LEN = 5000

replay_buffer = Queue(maxsize=REPLAY_BUFFER_LEN)
for ep in range(200):

    entry_state = env.reset()
    last_observations = Queue(maxsize=4)

    while True:

        env.render()

        if last_observations.full():
            last_observations.get()

        last_observations.put(entry_state)
        action = 0

        if randint(1, 10) == 5:
            action = env.action_space.sample()

        else:

            action = predict_action(last_observations)
        
        next_state, reward, done, info = env.step(action)

        #put transition to exp. replay
        transition = (last_observations, action, reward, next_state)

        if replay_buffer.full():
            replay_buffer.get()
        replay_buffer.put(transition)

        #calculate loss and update weights
        if done:
            print("episode done, reward: ", reward)
            break


env.close()

