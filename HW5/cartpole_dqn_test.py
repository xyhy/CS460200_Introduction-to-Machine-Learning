import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

EPISODE_LENGTH = 300

def get_hidden_nodes():
    with open('cartpole_dqn.json') as f:
        data = json.load(f)

    return int(data['num_hidden_nodes'])

# Basic Q-netowrk
class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        # Two fully-connected layers, input (state) to hidden & hidden to output (action)
        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_values = self.out(x)
        return action_values


# Deep Q-Network, composed of one eval network, one target network
class DQN(object):
    def __init__(self, n_states, n_actions, n_hidden, test = False):
        self.eval_net = Net(n_states, n_actions, n_hidden)

        if test:
            self.load_model()

    def choose_action(self, state):
        x = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)

        action_values = self.eval_net(x) # feed into eval net, get scores for each action
        action = torch.argmax(action_values).item() # choose the one with the largest score

        return action

    def load_model(self):
        self.eval_net.load_state_dict(torch.load('cartpole_dqn_model'))

env = gym.make('CartPole-v0')

# Environment parameters
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Hyper parameters
n_hidden = get_hidden_nodes()

# Create DQN
dqn = DQN(n_states, n_actions, n_hidden, True)

state = env.reset() # reset environment to initial state for each episode

for t in range(EPISODE_LENGTH):
    env.render()

    # Agent takes action
    action = dqn.choose_action(state) # choose an action based on DQN
    state, actual_reward, done, info = env.step(action) # do the action, get the reward

    if done:
        if t + 1 >= 200:
            print('Congradulation, you successfully pass the test!!!')
        else:
            print('Unfortunately, you need to work hard on your q table!!! {}',format(t + 1))

        break

env.close()