import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import gym
import pdb
import random
from collections import deque



class ExperienceReplay():
    def __init__(self, size):
        self.size = size 
        self.memory = deque(maxlen = size)

    def push(self, x):
        self.memory.append(x)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def performUpdates(self, lossF, optim, batch_size, Qnet, gamma):
        miniBatch = self.sample(batch_size)
        # pdb.set_trace()
        for i in range(0, batch_size):  
            sarsd = miniBatch[i]
            state = sarsd[0]
            action = sarsd[1]
            ri = sarsd[2]
            ns = sarsd[3]
            done = sarsd[4]
            QvalsForState = Qnet(get_variable_from_input(state))
            targetValForState = torch.FloatTensor()
            targetValForState = QvalsForState.data.clone()
            if done: 
                targetValForState[action] = ri
            else: 
                QvalForNextState = Qnet(get_variable_from_input(ns))
                maxQAction = torch.max(QvalForNextState)
                # pdb.set_trace()
                targetValForState[action] = (ri + gamma*maxQAction).data[0]

            optim.zero_grad()
            loss = lossF(QvalsForState, get_variable_from_input(targetValForState, False))
            loss.backward()
            optim.step()




class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        hiddenLayer = 30
        intermediateLayer = 25
        actionStates = 2
        self.fc1 = nn.Linear(4, hiddenLayer)
        self.fc2 = nn.Linear(hiddenLayer, 25)
        self.fc3 = nn.Linear(25, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)



#taken from pytorch tutorial 
def plot_durations(episode_durations):
    plt.figure(2)
    plt.clf()
    durations_t = torch.FloatTensor(episode_durations)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(durations_t.numpy())# Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_variable_from_input(x, grad=True):
    return Variable(torch.FloatTensor(x), requires_grad=grad)


def train(Qnet, optim, lossF, env, batch_size, gamma, total_episode, epsilon, min_epsilon, decay):
    experienceReplay = ExperienceReplay(50000)
    episodeTotalReward = []
    debug = True


    for episode in range(0, total_episode):
        state = env.reset()
        totalRewards = 0;

        for t in count(1):
            actionTaken = None
            if np.random.rand() <= epsilon: 
                actionTaken = np.random.randint(0, 2)
            else: 
                Qval = Qnet(get_variable_from_input(state))
                # pdb.set_trace()
                actionTaken = np.argmax(Qval.data.numpy())
            # pdb.set_trace()
            nextstate, reward, done, _ = env.step(actionTaken)

            if done: 
                reward = -10

            sarsd = (state, actionTaken, reward, nextstate, done)
            experienceReplay.push(sarsd)
            totalRewards=totalRewards+reward
            state = nextstate
            if done: 
                break

        if debug: 
            episodeTotalReward.append(totalRewards)
            plot_durations(episodeTotalReward)

        if episode > batch_size and episode > 0: 
            experienceReplay.performUpdates(lossF, optim, batch_size, Qnet, gamma)
  
            if epsilon > min_epsilon:
                epsilon = epsilon*0.995




def main():
    env = gym.make("CartPole-v0")
    #define neural net that outputs a stochastic policy 
    #for cartpole and for which we are trying to get the best 
    #policy
    Qnet = QNet()

    #define hyperparameters 
    batch_size = 100 
    gamma = 0.96
    total_episode = 2000 
    epsilon = 0.99
    minEpsilon = 0.1
    decay = 0.05 

    #define optimizer that wil
    optimizer = torch.optim.Adam(Qnet.parameters(), lr =0.001)
    loss = nn.MSELoss()

    train(Qnet, optimizer, loss, env, batch_size, gamma, total_episode, epsilon, minEpsilon, decay)


if __name__ == '__main__':
    main()
