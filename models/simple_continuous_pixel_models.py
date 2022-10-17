#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 17:55:49 2021

@author: vittorio
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class TanhGaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, use_memory = False, use_bn=True):
        super(TanhGaussianActor, self).__init__()
        
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.high = max_action
        self.low = -max_action
        
        # Decide which components are enabled
        self.use_memory = use_memory
        self.in_channel = state_dim[2]
        n = state_dim[0]
        m = state_dim[1]
        print(f"image dim:{n}x{m}")

        if use_bn:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 8, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, 16, (2, 2)),
                nn.ReLU(),
                nn.BatchNorm2d(16),
            )
            
        else:
            self.image_conv = nn.Sequential(
                nn.Conv2d(3, 8, (2, 2)),
                nn.ReLU(),
                nn.MaxPool2d((2, 2)),
                nn.Conv2d(8, 16, (2, 2)),
                nn.ReLU(),
                nn.Conv2d(16, 16, (2, 2)),
                nn.ReLU(),
            )
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*16
        print("Image embedding size: ", self.image_embedding_size)
        self.embedding = self.image_embedding_size
        
        # Define memory
        if self.use_memory:
            self.semi_memory_size = self.image_embedding_size
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            self.embedding += self.semi_memory_size
        
        self.reg_layer = nn.Linear(self.embedding, 64)
        
        self.actor = nn.Sequential(
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Define critic's model
        self.value_function = nn.Sequential(
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Define critic's model
        self.Q1 = nn.Sequential(
            nn.Linear(64+action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        # Define critic's model
        self.Q2 = nn.Sequential(
            nn.Linear(64+action_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )
        
        self.critic_target_Q1 = copy.deepcopy(self.Q1)
        self.critic_target_Q2 = copy.deepcopy(self.Q2)
        self.actor_target = copy.deepcopy(self.actor)
                    
        # Initialize parameters correctly
        self.apply(init_params)
        
    def encode_image(self, state):
        x = state 
        x = self.image_conv(x)
        embedding = x.reshape(x.shape[0], -1)
        bot = self.reg_layer(embedding)
        return bot
        
    def forward(self, state):
        embedding = self.encode_image(state)
        mean = self.actor(embedding)
        log_std = self.log_std.clamp(-20,2)
        std = torch.exp(log_std)
        return mean, std
    
    def unsquash(self, values):
        normed_values = (values - self.low[0])/(self.high[0] - self.low[0])*2.0 - 1.0
        stable_normed_values = torch.clamp(normed_values, -1+1e-4, 1-1e-4)
        unsquashed = torch.atanh(stable_normed_values)
        return unsquashed.float()
    
    def sample_log(self, state, action):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x = self.unsquash(action)
        y = torch.tanh(x) 
        log_prob = torch.clamp(normal.log_prob(x), -5, 5)
        log_prob -= torch.log((1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return log_prob
    
    def sample(self, state):
        mean, std = self.forward(state)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)        
        action = y*self.high[0]
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(self.high[0]*(1 - y.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)*self.high[0]
        return action, log_prob, mean 
    
    def Distb(self, state):
        embedding = self.encode_image(state)
        mean = self.actor(embedding)
        log_std = self.log_std.clamp(-20,2)
        std = torch.exp(log_std)
        cov_mtx = torch.eye(self.action_dim).to(device) * (std ** 2)
        distb = torch.distributions.MultivariateNormal(mean, cov_mtx)
        return distb
    
    def actor_target_net(self, state):
        embedding = self.encode_image(state)
        mean = self.actor_target(embedding)
        return self.high[0]*torch.tanh(mean)
    
    def deterministic_actor(self, state):
        embedding = self.encode_image(state)
        mean = self.actor(embedding)
        return self.high[0]*torch.tanh(mean)
    
    def value_net(self, state):
        embedding = self.encode_image(state)
        q1 = self.value_function(embedding)   
        return q1
    
    def critic_net(self, state, action):
        embedding = self.encode_image(state)
        sa = torch.cat([embedding, action], 1)
        q1 = self.Q1(sa)   
        q2 = self.Q2(sa)
        return q1, q2
    
    def critic_target(self, state, action):
        embedding = self.encode_image(state)
        sa = torch.cat([embedding, action], 1)
        q1 = self.critic_target_Q1(sa)   
        q2 = self.critic_target_Q2(sa)
        return q1, q2
        
            