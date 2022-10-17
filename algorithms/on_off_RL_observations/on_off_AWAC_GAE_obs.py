#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:54:01 2021

@author: vittorio
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.minigrid_models import InverseModels, SoftmaxActor
from models.minigrid_models import SoftmaxCritic
from models.minigrid_models import Encoder
from models.minigrid_models import Discriminator_GAN, Discriminator_WGAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class on_off_AWAC_GAE_obs:
    def __init__(self, state_dim, action_dim, action_space_cardinality, max_action, min_action, Domain_adaptation = True, Entropy = True, Train_encoder=True,   
                 num_steps_per_rollout=2000, intrinsic_reward = 0.01, number_obs_off_per_traj=100, l_rate_actor=3e-4, l_rate_alpha=3e-4, 
                 discount=0.99, tau=0.005, beta=3, gae_gamma = 0.99, gae_lambda = 0.99, minibatch_size=64, num_epochs=10, alpha=0.2, 
                 adversarial_loss = "wgan"):
        
        self.encoder_on = Encoder(state_dim).to(device)
        self.actor = SoftmaxActor(state_dim, action_space_cardinality, self.encoder_on.repr_dim).to(device)
        self.critic = SoftmaxCritic(state_dim, action_space_cardinality, self.encoder_on.repr_dim).to(device)
        self.inverse_models = InverseModels(state_dim, action_space_cardinality, self.encoder_on.repr_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=l_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=l_rate_actor)
        self.encoder_optimizer = torch.optim.Adam(self.encoder_on.parameters(), lr=l_rate_actor)
        self.inverse_models_optimizer = torch.optim.Adam(self.inverse_models.parameters(), lr=l_rate_actor)
        self.action_space = "Discrete"
        self.Train_encoder = Train_encoder
        
        if adversarial_loss == "gan":
            #encoder off-policy data
            self.encoder_off = copy.deepcopy(self.encoder_on)
            self.encoder_off_optimizer = torch.optim.Adam(self.encoder_off.parameters(), lr=2e-4, betas=(0.5, 0.999))
            
            # discriminator for domain adaptation
            self.discriminator = Discriminator_GAN(self.encoder_on.repr_dim).to(device)
            self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
            
        elif adversarial_loss == "wgan":
            #encoder off-policy data
            self.encoder_off = copy.deepcopy(self.encoder_on)
            self.encoder_off_optimizer = torch.optim.RMSprop(self.encoder_off.parameters(), lr=5e-5)
            
            # discriminator for domain adaptation
            self.discriminator = Discriminator_WGAN(self.encoder_on.repr_dim).to(device)
            self.discriminator_optimizer = torch.optim.RMSprop(self.discriminator.parameters(), lr=5e-5)
                
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_cardinality = action_space_cardinality
        self.max_action = max_action
        
        self.num_steps_per_rollout_on = num_steps_per_rollout
        self.discount = discount
        self.tau = tau
        self.beta = beta
        self.gae_gamma = gae_gamma
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        self.reward = []
        
        self.Entropy = Entropy
        self.target_entropy = -torch.FloatTensor([action_dim]).to(device)
        self.log_alpha = torch.zeros(1, requires_grad=True, device="cuda")
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = l_rate_alpha) 
        self.alpha = alpha
        
        self.Total_t = 0
        self.Total_iter = 0
        
        self.number_obs_off_per_traj = int(number_obs_off_per_traj)
        self.intrinsic_reward = intrinsic_reward
        
        self.Domain_adaptation = Domain_adaptation
        self.adversarial_loss = adversarial_loss
        
    def reset_counters(self):
        self.Total_t = 0
        self.Total_iter = 0
                
    def select_action(self, state):
        with torch.no_grad():
            if self.action_space == "Discrete":
                state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
                embedding = self.encoder_on(state)
                action, _ = self.actor.sample(embedding)
                return int((action).cpu().data.numpy().flatten())
        
    def GAE(self, env, args):
        step = 0
        self.Total_iter += 1
        self.states = []
        self.actions = []
        self.returns = []
        self.advantage = []
        self.gammas = []
        self.reward = []
        
        while step < self.num_steps_per_rollout_on: 
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_gammas = []
            episode_lambdas = []    
            state, done = env.reset(), False
            t=0
            episode_reward = 0

            while not done and step < self.num_steps_per_rollout_on: 
            # Select action randomly or according to policy
                if self.Total_t < args.start_timesteps:
                    if args.action_space == "Continuous":
                        action = env.action_space.sample() 
                    elif args.action_space == "Discrete":
                        action = env.action_space.sample()  
                else:
                    action = on_off_AWAC_GAE_obs.select_action(self, state)
            
                self.states.append(state.transpose(2,0,1))
                self.actions.append(action)
                episode_states.append(state.transpose(2,0,1))
                episode_actions.append(action)
                episode_gammas.append(self.gae_gamma**t)
                episode_lambdas.append(self.gae_lambda**t)
                
                state, reward, done, _ = env.step(action)
                
                episode_rewards.append(reward)
            
                t+=1
                step+=1
                episode_reward+=reward
                self.Total_t += 1
                        
            if done: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {self.Total_t}, Iter Num: {self.Total_iter}, Episode T: {t} Reward: {episode_reward:.3f}")
                
            episode_states = torch.FloatTensor(np.array(episode_states)).to(device)
            
            if self.action_space == "Discrete":
                episode_actions = torch.LongTensor(np.array(episode_actions)).to(device)
            elif self.action_space == "Continuous":
                episode_actions = torch.FloatTensor(np.array(episode_actions)).to(device)
            
            episode_rewards = torch.FloatTensor(np.array(episode_rewards)).to(device)
            episode_gammas = torch.FloatTensor(np.array(episode_gammas)).to(device)
            episode_lambdas = torch.FloatTensor(np.array(episode_lambdas)).to(device)
                
            episode_discounted_rewards = episode_gammas*episode_rewards
            episode_discounted_returns = torch.FloatTensor([sum(episode_discounted_rewards[i:]) for i in range(t)]).to(device)
            episode_returns = episode_discounted_returns
            self.returns.append(episode_returns)
            self.reward.append(episode_rewards)
            
            self.actor.eval()
            
            with torch.no_grad():
                embedding = self.encoder_on(episode_states)
                current_values = self.critic.value_net(embedding).detach()
                next_values = torch.cat((self.critic.value_net(embedding)[1:], torch.FloatTensor([[0.]]).to(device))).detach()
                episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:t-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(t)])
            
            self.advantage.append(episode_advantage)
            self.gammas.append(episode_gammas)
            
    def GAE_off(self, off_policy_data, ntrajs):
        states = []
        actions = []
        returns = []
        advantage = []
        gammas_list = []
        lambdas_list = []
        
        size_off_policy_data = len(off_policy_data)
        ind = np.random.randint(0, size_off_policy_data-self.number_obs_off_per_traj-1, size=ntrajs)
        
        sampled_states = []
        sampled_actions = []
        sampled_rewards = []
        
        with torch.no_grad():
            
            self.actor.eval()
        
            for i in range(ntrajs):
                states_temp = torch.FloatTensor(off_policy_data[ind[i]:int(ind[i]+self.number_obs_off_per_traj)]).to(device)
                next_states_temp = torch.FloatTensor(off_policy_data[int(ind[i]+1):int(ind[i]+self.number_obs_off_per_traj+1)]).to(device)
                embedding = self.encoder_off(states_temp)
                embedding_next = self.encoder_off(next_states_temp)
                
                actions_temp = self.inverse_models.sample_inverse_model(embedding, embedding_next)
                
                rewards_temp = self.inverse_models.forward_inv_reward(embedding, embedding_next)
                rewards_i = self.intrinsic_reward*torch.ones_like(rewards_temp)    
                rewards_tot = rewards_temp + rewards_i
            
                sampled_states.append(states_temp)
                sampled_actions.append(actions_temp)
                sampled_rewards.append(rewards_tot)
            
            for l in range(ntrajs):
                traj_size = self.number_obs_off_per_traj
                gammas = []
                lambdas = []
                for t in range(traj_size):
                    gammas.append(self.gae_gamma**t)
                    lambdas.append(self.gae_lambda**t)
                    
                gammas_list.append(torch.FloatTensor(np.array(gammas)).to(device))
                lambdas_list.append(torch.FloatTensor(np.array(lambdas)).to(device))
                    
            for l in range(ntrajs):
                
                episode_states = sampled_states[l]
                episode_actions = sampled_actions[l]
                episode_rewards = sampled_rewards[l].squeeze() 
                episode_gammas = gammas_list[l]
                episode_lambdas = lambdas_list[l]
                
                traj_size = self.number_obs_off_per_traj
     
                episode_discounted_rewards = episode_gammas*episode_rewards
                episode_discounted_returns = torch.FloatTensor([episode_discounted_rewards[i:].sum() for i in range(traj_size)]).to(device)
                episode_returns = episode_discounted_returns
                
                embedding = self.encoder_off(episode_states)
                current_values = self.critic.value_net(embedding).detach()
                next_values = torch.cat((self.critic.value_net(embedding)[1:], torch.FloatTensor([[0.]]).to(device))).detach()
                episode_deltas = episode_rewards.unsqueeze(-1) + self.gae_gamma*next_values - current_values
                episode_advantage = torch.FloatTensor([((episode_gammas*episode_lambdas)[:traj_size-j].unsqueeze(-1)*episode_deltas[j:]).sum() for j in range(traj_size)]).to(device)
                
                states.append(episode_states)
                actions.append(episode_actions)
                returns.append(episode_returns)
                advantage.append(episode_advantage)
            
        return states, actions, returns, advantage
    
    def train_inverse_models(self):
        states_on = torch.FloatTensor(np.array(self.states)).to(device)
        
        actions_on = torch.LongTensor(np.array(self.actions)).to(device)
        
        reward_on = torch.cat(self.reward)
        
        max_steps = self.num_epochs * (self.num_steps_per_rollout_on // self.minibatch_size)
        
        for _ in range(max_steps):
            
            minibatch_indices_ims = np.random.choice(range(self.num_steps_per_rollout_on-1), self.minibatch_size, False)
            states_ims = states_on[minibatch_indices_ims]
            next_states_ims = states_on[minibatch_indices_ims+1]
            rewards_ims = reward_on[minibatch_indices_ims]
            actions_ims = actions_on[minibatch_indices_ims]
            
            with torch.no_grad():
                embedding = self.encoder_on(states_ims)
                embedding_next = self.encoder_on(next_states_ims)
            
            inverse_action_model_prob = self.inverse_models.forward_inv_a(embedding, embedding_next)
            m = F.one_hot(actions_ims.squeeze().cpu(), self.action_space_cardinality).float().to(device)
            L_ia = F.mse_loss(inverse_action_model_prob, m)
            
            L_ir = F.mse_loss(rewards_ims.unsqueeze(-1), self.inverse_models.forward_inv_reward(embedding, embedding_next))
              
            self.inverse_models_optimizer.zero_grad()
            loss = L_ia + L_ir 
            loss.backward()
            self.inverse_models_optimizer.step()
    
    def train(self, states_off, actions_off, returns_off, advantage_off):
        
        states_on = torch.FloatTensor(np.array(self.states)).to(device)
        actions_on = torch.LongTensor(np.array(self.actions)).to(device)
        
        returns_on = torch.cat(self.returns)
        advantage_on = torch.cat(self.advantage).to(device)
        
        states_off = torch.cat(states_off)
        actions_off = torch.cat(actions_off).squeeze()
        
        returns_off = torch.cat(returns_off)
        
        advantage_off = torch.cat(advantage_off)

        advantage_on = (advantage_on-advantage_on.mean())/(advantage_on.std()+1e-6)
        advantage_off = (advantage_off-advantage_off.mean())/(advantage_off.std()+1e-6)
        
        self.actor.train()
        
        obs_off_size = len(advantage_off)
        self.num_steps_per_rollout = len(advantage_on)
        max_steps = self.num_epochs * (self.num_steps_per_rollout // self.minibatch_size)
        
        recap_d_loss = []
        recap_encode_loss = []
        recap_real_score = []
        recap_fake_score = []
        
        for i in range(max_steps):
            
            minibatch_indices = np.random.choice(range(self.num_steps_per_rollout), self.minibatch_size, False)
            minibatch_indices_off = np.random.choice(range(obs_off_size), self.minibatch_size, False)

            batch_actions_on = actions_on[minibatch_indices]
            batch_returns_on = returns_on[minibatch_indices]
            batch_advantage_on = advantage_on[minibatch_indices]
            batch_states_on = states_on[minibatch_indices]

            batch_actions_off = actions_off[minibatch_indices_off]
            batch_returns_off = returns_off[minibatch_indices_off]
            batch_advantage_off = advantage_off[minibatch_indices_off]
            batch_states_off = states_off[minibatch_indices_off]

            batch_embedding_on = self.encoder_on(batch_states_on)
            batch_embedding_off = self.encoder_off(batch_states_off)

            L_vf_on = ((self.critic.value_net(batch_embedding_on).squeeze() - batch_returns_on)**2).mean()
            L_vf_off = ((self.critic.value_net(batch_embedding_off).squeeze() - batch_returns_off)**2).mean()
        
            if self.Train_encoder:
                self.encoder_off_optimizer.zero_grad(set_to_none=True)
                self.encoder_optimizer.zero_grad(set_to_none=True)

            self.critic_optimizer.zero_grad(set_to_none=True)
            (L_vf_on + L_vf_off).backward()

            if self.Train_encoder:
                self.encoder_off_optimizer.step()
                self.encoder_optimizer.step()

            self.critic_optimizer.step()

            batch_embedding = torch.cat((batch_embedding_on, batch_embedding_off))
            batch_actions = torch.cat((batch_actions_on, batch_actions_off))
            batch_advantage = torch.cat((batch_advantage_on, batch_advantage_off))
                    
            log_prob, log_prob_rollout = self.actor.sample_log(batch_embedding.detach(), batch_actions)
                
            r = (log_prob_rollout).squeeze()
            weights = F.softmax(batch_advantage/self.beta, dim=0).detach()
            L_clip = r*weights
            
            self.actor_optimizer.zero_grad()
            if self.Entropy:
                _, log_pi_state = self.actor.sample(batch_embedding.detach())
                    
                loss = (-1) * (L_clip - self.alpha*log_pi_state).mean()
            else:
                loss = (-1) * (L_clip).mean()
            
            loss.backward()
            self.actor_optimizer.step()
            
            if self.Domain_adaptation:
                
                minibatch_indices_ims = np.random.choice(range(self.num_steps_per_rollout_on), self.minibatch_size, False)
                states_ims = states_on[minibatch_indices_ims]
                
                minibatch_indices_obs = np.random.choice(range(obs_off_size), self.minibatch_size, False)
                state_obs = states_off[minibatch_indices_obs]
                
                if self.adversarial_loss == 'gan':
                    
                    obs_class = torch.zeros(self.minibatch_size, device=device)
                    rollout_class = torch.ones(self.minibatch_size, device=device)
                    criterion = torch.nn.BCELoss()
                    
                    # -----------------
                    #  Train Encoder off
                    # -----------------
                    
                    self.encoder_off_optimizer.zero_grad()
                    
                    embedding_off = self.encoder_off(state_obs)
                    encode_loss = criterion(self.discriminator(embedding_off).squeeze(), rollout_class)
    
                    encode_loss.backward()
                    self.encoder_off_optimizer.step()
                    
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    
                    self.discriminator_optimizer.zero_grad()
                    
                    real_embedding = self.encoder_on(states_ims).detach()
                    d_loss_rollout = criterion(self.discriminator(real_embedding).squeeze(), rollout_class) 
                    d_loss_obs = criterion(self.discriminator(embedding_off.detach()).squeeze(), obs_class) 
                    d_loss = 0.5*(d_loss_rollout + d_loss_obs)
                    
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                    
                elif self.adversarial_loss == 'wgan':
                    
                    clip_value = 0.01
                    n_critic = 5
                    
                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    
                    self.discriminator_optimizer.zero_grad()
                    
                    embedding_off = self.encoder_off(state_obs).detach().squeeze()
                    real_embedding = self.encoder_on(states_ims).detach().squeeze()
                    
                    d_loss = -torch.mean(self.discriminator(real_embedding)) + torch.mean(self.discriminator(embedding_off))
                    
                    d_loss.backward()
                    self.discriminator_optimizer.step()
                    
                    # Clip weights of discriminator
                    for p in self.discriminator.parameters():
                        p.data.clamp_(-clip_value, clip_value)
                        
                    # Train the encoder off every n_critic iterations
                    if i % n_critic == 0:
                        
                        # -----------------
                        #  Train Encoder off
                        # -----------------
                        
                        self.encoder_off_optimizer.zero_grad()
                        
                        embedding_off = self.encoder_off(state_obs).squeeze()
                        encode_loss = -torch.mean(self.discriminator(embedding_off))
        
                        encode_loss.backward()
                        self.encoder_off_optimizer.step()
                        
                else:
                    NotImplemented
                    
                recap_d_loss.append(d_loss.detach().cpu().numpy())
                recap_encode_loss.append(encode_loss.detach().cpu().numpy())
                recap_real_score.append((self.discriminator(self.encoder_on(states_ims).detach())).mean().detach().cpu().numpy())
                recap_fake_score.append((self.discriminator(self.encoder_off(state_obs).detach())).mean().detach().cpu().numpy())
                    
            if self.Entropy: 

                alpha_loss = -(self.log_alpha * (log_pi_state + self.target_entropy).detach()).mean()
        
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
        
                self.alpha = self.log_alpha.exp()
                
        if self.Domain_adaptation:
        
            mean_d = np.mean(np.array(recap_d_loss))
            mean_encode = np.mean(np.array(recap_encode_loss))
            real_score = np.mean(np.array(recap_real_score))
            fake_score = np.mean(np.array(recap_fake_score))
            
            print(f"d_loss: {mean_d}, encode_loss: {mean_encode}")
            print(f"class real score: {real_score}, class fake score: {fake_score}")
                
    def save_actor(self, filename):
        option = 0
        torch.save(self.actor.state_dict(), filename + f"_pi_lo_option_{option}")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_pi_lo_optimizer_option_{option}")
    
    def load_actor(self, filename):
        option = 0
        self.actor.load_state_dict(torch.load(filename + f"_pi_lo_option_{option}"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_pi_lo_optimizer_option_{option}"))

    def save_critic(self, filename):
        torch.save(self.value_function.state_dict(), filename + "_value_function")
        torch.save(self.value_function_optimizer.state_dict(), filename + "_value_function_optimizer")
    
    def load_critic(self, filename):
        self.value_function.load_state_dict(torch.load(filename + "_value_function"))      
        self.value_function_optimizer.load_state_dict(torch.load(filename + "_value_function_optimizer")) 
        
        
        
        

            
            
        
            
            
            

        