
import gym
import json
import math
import numpy as np

# Training episode setting
EPISODE_LENGTH = 300

def get_qtable():
    with open('cartpole_qtable.json') as f:
        data = json.load(f)

    return np.asarray(data['qtable'])

def get_buckets():
    with open('cartpole_qtable.json') as f:
        data = json.load(f)
        
    return tuple(data['buckets'])

def choose_action(state, q_table):
    return np.argmax(q_table[state])

def get_state(observation, n_buckets, state_bounds):
    state = [0] * len(observation)
    for i, s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # lower- and upper-bounds for each feature in observation
        if s <= l:
            state[i] = 0
        elif s >= u:
            state[i] = n_buckets[i] - 1
        else:
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)

# Environment setting
env = gym.make('CartPole-v0')

n_buckets = get_buckets()     
q_table = get_qtable()

## state bounds
state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

epsilon = 1.0

observation = env.reset()   # reset environment to initial state for each episode
state = get_state(observation, n_buckets, state_bounds) # turn observation into discrete state

for t in range(EPISODE_LENGTH):
    env.render()

    # Agent takes action
    action = choose_action(state, q_table)                      # choose an action based on q_table 
    observation, reward, done, info = env.step(action)          # do the action, get the reward
    state = get_state(observation, n_buckets, state_bounds)

    if done:
        if t + 1 >= 200:
            print('Congradulation, you successfully pass the test!!!')
        else:
            print('Unfortunately, you need to work hard on your q table!!! {}',format(t + 1))

        break

env.close() # need to close, or errors will be reported