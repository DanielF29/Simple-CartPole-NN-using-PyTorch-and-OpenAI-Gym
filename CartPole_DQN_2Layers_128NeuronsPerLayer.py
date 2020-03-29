#This code is base on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html and on the adaptation of this code on https://discuss.pytorch.org/t/help-for-dqn-example/13263
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable

########################################################################################################

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, *args):
        #"""Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

########################################################################################################

class DQN_model(nn.Module):
    def __init__(self, num_state):
        super(DQN_model, self).__init__()
        self.model = nn.Sequential(
                                   torch.nn.Linear(num_state, hl_size*2),  #in:4,  out:128
                                   nn.LeakyReLU(0.2, inplace=True),
                                   torch.nn.Linear(hl_size*2, n_actions),  #in:128, out:2
                                   nn.LeakyReLU(0.2, inplace=True)         
                                   )
    def forward(self, x):
        return self.model(x)
    
########################################################################################################    

def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
########################################################################################################
########################################################################################################

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            out_policy_net_aux = policy_net(state)
            out_policy_net_aux = out_policy_net_aux.view(1, 2)
            return out_policy_net_aux.max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

########################################################################################################

def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.000001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

########################################################################################################

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = [s.view(1,-1) for s in batch.state]
    state_batch = torch.cat(state_batch, 0)
    state_batch = Variable(state_batch)

    action_batch = torch.cat(batch.action)
    action_batch = Variable(action_batch) 
    
    reward_batch = torch.cat(batch.reward)
    reward_batch = Variable(reward_batch)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters(): 
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    

def CartPole_():
  global steps_done, hl_size, num_state, n_actions, EPS_END, BATCH_SIZE, GAMMA, EPS_START, EPS_DECAY, TARGET_UPDATE, Transition, episode_durations, device, memory, is_ipython, policy_net, target_net, loss_fn, optimizer
  env = gym.make('CartPole-v0').unwrapped
  num_state = env.reset()
  num_state = len(num_state) #these last 2 lines to define the number of observations got from an state of the simulation (which is 4), 
                           #and we will use it as the number of input for our Neural-Net
  
  # set up matplotlib
  is_ipython = 'inline' in matplotlib.get_backend()
  if is_ipython:
      from IPython import display
  plt.ion()

  # if gpu is to be used
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
  Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

  BATCH_SIZE = 64 #4 #16 #8 #64
  GAMMA = 0.999
  EPS_START = 0.9
  EPS_END = 0.05
  EPS_DECAY = 1000 #200 t was recomended a slower decay here: https://discuss.pytorch.org/t/help-for-dqn-example/13263
  TARGET_UPDATE = 10
  hl_size = 64 #Number of Neurons on each layer
  
  # Get number of actions from gym action space
  n_actions = 2 #Just in case here the number of possible actions is been fixed instead of calculating it with: env.action_space.n
  steps_done = 0
  episode_durations = []

  policy_net = DQN_model(num_state).to(device).double()
  policy_net.apply(weights_init_normal)
  target_net = DQN_model(num_state).to(device).double()
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()
  
  #optimizer = optim.Adam(policy_net.parameters(),  lr=1e-4)
  optimizer = optim.RMSprop(policy_net.parameters()) #It works better with this than with Adam optimizer
  #loss_fn = nn.MSELoss()
  #loss_fn = nn.NLLLoss()
  loss_fn = nn.SmoothL1Loss() 
  
  memory = ReplayMemory(2000) #500) #1000)
  


  num_episodes = 500 #1000 #10000
  for i_episode in range(num_episodes):
      # Initialize the environment and state
      state = env.reset() 
      state = torch.from_numpy(state)
      for t in count():
      #for t in range(num_episodes):
          state = state.to(device)
          action = select_action(state.double())
          next_state, reward, done, info = env.step(action.item())
          
          #pass reward as a tensor
          reward = torch.tensor([reward], device=device) 
          
          # Observe new state
          next_state= torch.tensor([next_state], device=device)
          next_state = next_state if not done else None
          
          # Store the transition in memory
          assert state is not None
          memory.push(state, action, next_state, reward) 
          
          # Move to the next state
          state = next_state
          
          # Perform one step of the optimization (on the target network)
          optimize_model()
          #env.render() #Commenting this line hopping to do a faster training
          if done:
              episode_durations.append(t + 1)
              plot_durations()
              break
          
      # Update the target network, copying all weights and biases in DQN
      if i_episode % TARGET_UPDATE == 0:
          target_net.load_state_dict(policy_net.state_dict())

  print('Complete')
  env.render()
  env.close()
  plt.ioff()
  plt.show()


if __name__ == '__main__':
    CartPole_()

    
    