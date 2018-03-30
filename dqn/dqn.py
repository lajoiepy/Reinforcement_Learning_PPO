#!/usr/bin/python3

# based on http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
import random
from collections import namedtuple

import gym
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class DQN(nn.Module):
    def __init__(self, action_space_count):
        super().__init__()
        self.fc1 = nn.Linear(4, 512)
        self.fc2 = nn.Linear(512, 512)
        self.advantage_fc = nn.Linear(512, action_space_count)
        self.value_fc = nn.Linear(512, 1)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.advantage_fc(x)
        return q


class DDQN(nn.Module):
    def __init__(self, action_space_count):
        super().__init__()
        self.fc1 = nn.Linear(2, 512)
        self.fc2 = nn.Linear(512, 512)

        self.fc1_adv = nn.Linear(2, 512)
        self.fc2_adv = nn.Linear(512, 512)

        self.advantage_fc = nn.Linear(512, action_space_count)
        self.value_fc = nn.Linear(512, 1)

        self.action_space_count = action_space_count

    def forward(self, x):
        in_size = x.size(0)
        x2 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_fc(x)

        x2 = F.relu(self.fc1_adv(x2))
        x2 = F.relu(self.fc2_adv(x2))
        advantage = self.advantage_fc(x2)

        q = value + advantage - advantage.mean(2).unsqueeze(2).expand(in_size, 1, self.action_space_count)
        return q


# from https://github.com/ghliu/pytorch-ddpg/blob/master/util.py
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Agent(object):
    def __init__(self, action_space_count=2, gamma=0.8, agent_batch_size=128):
        self.target_Q = DDQN(action_space_count)
        self.Q = DDQN(action_space_count)
        self.gamma = gamma
        self.batch_size = agent_batch_size
        hard_update(self.target_Q, self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=0.0001)
        self.number_train = 0

    def act(self, x, environment, epsilon_param=0.1):
        rnd_number = np.random.rand(1)
        x = x.unsqueeze(0).unsqueeze(1)
        if rnd_number[0] > (1 - epsilon_param):
            action_to_select = Variable(torch.LongTensor([environment.action_space.sample()]))
            return action_to_select
        else:
            q = self.Q.forward(x)
            action_to_select = torch.max(q, 2)[1][0]
            return action_to_select

    def backward(self, transitions, nb_iter):
        current_batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.stack(current_batch.state))
        next_state_batch = Variable(torch.stack(current_batch.next_state), volatile=True)
        action_batch = Variable(torch.stack(current_batch.action).view(-1, 1, 1))
        reward_batch = Variable(torch.stack(current_batch.reward))
        done_batch = Variable(torch.stack(current_batch.done)).type(torch.LongTensor)

        q_value_prediction = self.Q(state_batch).gather(2, action_batch)

        # Double DQN
        next_state_action = self.Q(next_state_batch).max(2, keepdim=True)[1]
        target_q_values = self.target_Q(next_state_batch).detach().gather(2, next_state_action)

        # Sans double DQN
        # target_q_values = self.target_Q(next_state_batch).detach().max(2)[0]

        target_q_values_reward = Variable(torch.FloatTensor(len(current_batch.state), 1).zero_())
        for index in range(len(current_batch.state)):
            if done_batch.data[index][0] == 0:
                target_q_values_reward[index] = reward_batch[index] + (self.gamma * target_q_values[index])
            else:
                target_q_values_reward[index] = reward_batch[index]

        target_q_values_reward.volatile = False

        self.optimizer.zero_grad()
        loss = F.smooth_l1_loss(q_value_prediction, target_q_values_reward)
        loss.backward()
        self.optimizer.step()
        soft_update(self.target_Q, self.Q, 0.995)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, sample_batch_size):
        return random.sample(self.memory, sample_batch_size)

    def __len__(self):
        return len(self.memory)

gym.envs.register(
    id='MountainCarTheHardestVersionEver-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=10000,      # MountainCar-v0 uses 200
    reward_threshold=-110.0,
)
env = gym.make('MountainCarTheHardestVersionEver-v0')

batch_size = 128
agent = Agent(action_space_count=env.action_space.n, agent_batch_size=batch_size)
memory = ReplayMemory(100000)

epsilon = 1
rewards = []

for i in range(5000):
    obs = env.reset()
    done = False
    total_reward = 0
    epsilon *= 0.99
    while not done:
        epsilon = max(epsilon, 0.1)
        obs_input = Variable(torch.from_numpy(obs).type(torch.FloatTensor))
        action = agent.act(obs_input, env, epsilon)
        next_obs, reward, done, _ = env.step(action.data[0])
        memory.push(obs_input.data.view(1, -1), action.data,
                    torch.from_numpy(next_obs).type(torch.FloatTensor).view(1, -1), torch.Tensor([reward]),
                    torch.Tensor([done]))
        obs = next_obs
        total_reward += reward
    if (total_reward > -200):
        print("iteration {0} : moins de 200 steps (solved)".format(i))
    # print(i, total_reward, done)
    rewards.append(total_reward)
    if memory.__len__() > 10000:
        batch = memory.sample(batch_size)
        agent.backward(batch, i)

pd.DataFrame(rewards).rolling(50, center=False).mean().plot()
plt.show()

#plt.plot(rewards)
