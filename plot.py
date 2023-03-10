#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 17:44:56 2021

@author: vittorio
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as ptch
import pickle

def load_obj(name):
    with open('specs/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# environments = ['BipedalWalker-v3', 'BipedalWalkerHardcore-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2',
#                 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3',
#                 'FetchPickAndPlace-v1', 'FetchPush-v1', 'FetchReach-v1', 'FetchSlide-v1', 'HandManipulateBlock-v0',
#                 'HandManipulateEgg-v0', 'HandManipulatePen-v0', 'HandReach-v0']

environments = ['BipedalWalker-v3', 'LunarLander-v2', 'LunarLanderContinuous-v2', 'Ant-v3', 'HalfCheetah-v3', 'Hopper-v3', 
                'Humanoid-v3', 'HumanoidStandup-v2', 'Swimmer-v3', 'Walker2d-v3', 'BipedalWalkerHardcore-v3']

environments = ['BipedalWalkerHardcore-v3']

modes = ['RL', 'HRL']
RL_algorithms = ['PPO', 'TRPO', 'UATRPO', 'GePPO', 'TD3', 'SAC']
HRL = ['HPPO', 'HTRPO', 'HUATRPO', 'HGePPO', 'HTD3', 'HSAC']

colors = {}

colors['PPO'] = 'tab:blue'
colors['TRPO'] = 'tab:orange'
colors['UATRPO'] = 'tab:pink'
colors['GePPO'] = 'lime'
colors['TD3'] = 'tab:red'
colors['SAC'] = 'tab:purple'
colors['HPPO_2'] = 'tab:brown'
colors['HTRPO_2'] = 'tab:green'
colors['HUATRPO_2'] = 'tab:gray'
colors['HGePPO_2'] = 'chocolate'
colors['HTD3_2'] = 'tab:olive'
colors['HSAC_2'] = 'tab:cyan'
colors['HPPO_3'] = 'lightcoral'
colors['HTRPO_3'] = 'fuchsia'
colors['HUATRPO_3'] = 'gold'
colors['HGePPO_3'] = 'magenta'
colors['HTD3_3'] = 'lightseagreen'
colors['HSAC_3'] = 'peru'

# %% Classical Algorithms from the literature

# RL_algorithms = ['AWAC', 'AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 
#                   'GeA2C', 'GePPO', 'PPO', 'SAC', 'TD3']

RL_algorithms = ['PPO', 'AWAC_GAE', 'AWAC_Q_lambda_Peng']

charts = {}

colors = {}

Top_three = False

colors['AWAC'] = 'tab:blue'
colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['GeA2C'] = 'tab:green'
colors['GePPO'] = 'tab:gray'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['TD3'] = 'tab:cyan'

environments = ['MiniGrid-Empty-16x16-v0']

for env in environments:
    
    columns = 1
    rows = 1
    
    fig, ax = plt.subplots(rows, columns, figsize=(6,4))
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
    # fig.suptitle(env, fontsize="xx-large")
    
    for k, ax_row in enumerate([ax]):
        for j, axes in enumerate([ax_row]):
    
            for i in range(len(RL_algorithms)):
            
                policy = RL_algorithms[i]
                charts[policy]={}
                
                RL = []
                
                for seed in range(10):
                    
                    try:
                        if policy == "TD3" or policy == "SAC":
                            with open(f'results/RL/evaluation_RL_{policy}_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))  
                        else:
                            with open(f'results/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                                RL.append(np.load(f, allow_pickle=True))    
                                
                    except:
                        continue
                        
                try:
                    # if Top_three:
                    #     temp = []
                    #     RL_temp = []
                    #     for k in range(10):
                    #         temp.append([RL[k][-1], k])
                            
                    #     temp.sort()
                            
                    #     RL_temp.append(RL[temp[-1][1]])
                    #     RL_temp.append(RL[temp[-2][1]])
                    #     RL_temp.append(RL[temp[-3][1]])
                        
                    #     RL = RL_temp
                            
                    
                    mean = np.mean(np.array(RL), 0)
                    steps = np.linspace(0, (len(mean)-1)*2048, len(mean))
                    std = np.std(np.array(RL),0)
                    
                    charts[policy]['mean'] = mean[-1]
                    charts[policy]['std'] = std[-1]
                    
                    print(policy + f" {mean[-1]} +- {std[-1]}")
                    
                    axes.plot(steps, mean, label=policy, c=colors[policy])
                    axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                    
                    axes.set_ylim([0, 1.05])
                    # axes.set_xlim([0, 200000])
                    axes.set_xlabel('Frames')
                    axes.set_ylabel('Reward')
                except:
                    continue

    if not os.path.exists(f"./Figures/{env}"):
        os.makedirs(f"./Figures/{env}")
   
    handles, labels = axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=4, prop={'size': 12})
    plt.title(f'{env} Online RL')
           
    plt.savefig(f'Figures/{env}/{env}_Online_RL.pdf', format='pdf', bbox_inches='tight')


# %% On_off RL from human expert observations

# RL_algorithms = ['AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 'PPO', 'SAC', 'AWAC']
RL_algorithms = ['AWAC_GAE', 'AWAC_Q_lambda_Peng']

colors = {}

data_set = ['modified_human_expert', 'rodent']
domain_adaptation = ['True', 'False']
environments = ['MiniGrid-Empty-16x16-v0']
intrinsic_reward = ['0.01', '0.005']

colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['AWAC'] = 'tab:blue'


for env in environments:
    for domain_adapt in domain_adaptation:
        data = data_set[0]

        columns = 1
        rows = 1
        
        fig, ax = plt.subplots(rows, columns, figsize=(6,4))
        plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
        
        for k, ax_row in enumerate([ax]):
            for j, axes in enumerate([ax_row]):
        
                for i in range(len(RL_algorithms)):
                    
                    if j == 0:
                        Top_three = False
                    else:
                        Top_three = True
                
                    policy = RL_algorithms[i]
                    charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: 0.01'] = {}
                    
                    if policy == 'AWAC_GAE':
                        policy_label = 'AWPO+GAE'
                    elif policy == 'AWAC_Q_lambda_Haru':
                        policy_label = 'AWPO+HQL'
                    elif policy == 'AWAC_Q_lambda_Peng':
                        policy_label = 'AWPO+PQL'
                    elif policy == 'AWAC_Q_lambda_TB':
                        policy_label = 'AWPO+TBL'
                    elif policy == 'AWAC':
                        policy_label = 'AWAC'
                    elif policy == 'SAC':
                        policy_label = policy
                    elif policy == 'PPO':
                        policy_label = policy
                        
                    a_file = open(f"offline_data_set/data_set_{env}_human_expert.pkl", "rb")
                    offline_set = pickle.load(a_file)   
                    
                    reward_set = np.sum(offline_set['rewards'])/np.sum(offline_set['terminals'])
                    
                    RL = []
                    
                    for seed in range(10):
                        try:
                            if policy == "PPO":
                                with open(f'results/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                                    RL.append(np.load(f, allow_pickle=True))  
                            else:
                                if policy == "SAC":
                                    with open(f'results/on_off_RL_from_observations/evaluation_on_off_RL_from_observations_{policy}_{env}_dataset_{data}_domain_adaptation_{domain_adapt}_ri_0.01_{seed}.npy', 'rb') as f:
                                        RL.append(np.load(f, allow_pickle=True))     
                                else:
                                    with open(f'results/on_off_RL_from_observations/evaluation_on_off_RL_from_observations_{policy}_Entropy_False_{env}_dataset_{data}_domain_adaptation_{domain_adapt}_ri_0.01_{seed}.npy', 'rb') as f:
                                        RL.append(np.load(f, allow_pickle=True))    
                                    
                        except:
                            continue
                            
                    try:
                        
                        # if Top_three:
                        #     temp = []
                        #     RL_temp = []
                        #     for k in range(10):
                        #         try:
                        #             temp.append([RL[k][-1], k])
                        #         except:
                        #             continue
                                
                        #     temp.sort()
                                
                        #     RL_temp.append(RL[temp[-1][1]])
                        #     RL_temp.append(RL[temp[-2][1]])
                        #     RL_temp.append(RL[temp[-3][1]])
                            
                        #     RL = RL_temp
                        
                        mean = np.mean(np.array(RL), 0)
                        steps = np.linspace(0, (len(mean)-1)*2048, len(mean))
                        std = np.std(np.array(RL),0)
                        axes.plot(steps, mean, label=policy_label, c=colors[policy])
                        axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                        
                        axes.set_ylim([0, 1.05])
                        # axes.set_xlim([0, 200000])
                        axes.set_xlabel('Frames')
                        axes.set_ylabel('Reward')
                        
                        print(f'dataset: {data}, agent: {policy}, reward: {mean[-1]}, std: {std[-1]} , DA: {domain_adapt}')
                        charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: 0.01']['mean'] = mean[-1]
                        charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: 0.01']['std'] = std[-1]
                        
                    except:
                        continue
                    
                axes.plot(steps, reward_set*np.ones((len(steps))), 'k--', label='average reward data set')
    
        if not os.path.exists(f"./Figures/{env}"):
            os.makedirs(f"./Figures/{env}")
                  
        handles, labels = axes.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, prop={'size': 12})
        plt.title(f'{env} AWPO w/ Offline Observations')  
               
        plt.savefig(f'Figures/{env}/on_off_RL_from_observations_{env}_dataset_{data}_domain_adaptation_{domain_adapt}.pdf', format='pdf', bbox_inches='tight')
        
# %% On_off RL from rodent observations

# RL_algorithms = ['AWAC_GAE', 'AWAC_Q_lambda_Haru', 'AWAC_Q_lambda_Peng', 'AWAC_Q_lambda_TB', 'PPO', 'SAC', 'AWAC']
RL_algorithms = ['AWAC_GAE', 'AWAC_Q_lambda_Peng']

colors = {}

data_set = ['modified_human_expert', 'rodent']
domain_adaptation = ['True', 'False']
environments = ['MiniGrid-Empty-16x16-v0']
intrinsic_reward = ['0.01', '0.005']

colors['AWAC_GAE'] = 'tab:orange'
colors['AWAC_Q_lambda_Haru'] = 'lime'
colors['AWAC_Q_lambda_Peng'] = 'tab:purple'
colors['AWAC_Q_lambda_TB'] = 'tab:brown'
colors['PPO'] = 'chocolate'
colors['SAC'] = 'tab:olive'
colors['AWAC'] = 'tab:blue'


for env in environments:
    for domain_adapt in domain_adaptation:
        for ri in intrinsic_reward:
            data = data_set[1]
    
            columns = 1
            rows = 1
            
            fig, ax = plt.subplots(rows, columns, figsize=(6,4))
            plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.2)
            
            for k, ax_row in enumerate([ax]):
                for j, axes in enumerate([ax_row]):
            
                    for i in range(len(RL_algorithms)):
                        
                        if j == 0:
                            Top_three = False
                        else:
                            Top_three = True
                    
                        policy = RL_algorithms[i]
                        charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: {ri}'] = {}
                        
                        if policy == 'AWAC_GAE':
                            policy_label = 'AWPO+GAE'
                        elif policy == 'AWAC_Q_lambda_Haru':
                            policy_label = 'AWPO+HQL'
                        elif policy == 'AWAC_Q_lambda_Peng':
                            policy_label = 'AWPO+PQL'
                        elif policy == 'AWAC_Q_lambda_TB':
                            policy_label = 'AWPO+TBL'
                        elif policy == 'AWAC':
                            policy_label = 'AWAC'
                        elif policy == 'SAC':
                            policy_label = policy
                        elif policy == 'PPO':
                            policy_label = policy
                                                
                        RL = []
                        
                        for seed in range(10):
                            try:
                                if policy == "PPO":
                                    with open(f'results/RL/evaluation_RL_{policy}_Entropy_True_{env}_{seed}.npy', 'rb') as f:
                                        RL.append(np.load(f, allow_pickle=True))  
                                else:
                                    if policy == "SAC":
                                        with open(f'results/on_off_RL_from_observations/evaluation_on_off_RL_from_observations_{policy}_{env}_dataset_{data}_domain_adaptation_{domain_adapt}_ri_{ri}_{seed}.npy', 'rb') as f:
                                            RL.append(np.load(f, allow_pickle=True))     
                                    else:
                                        with open(f'results/on_off_RL_from_observations/evaluation_on_off_RL_from_observations_{policy}_Entropy_False_{env}_dataset_{data}_domain_adaptation_{domain_adapt}_ri_{ri}_{seed}.npy', 'rb') as f:
                                            RL.append(np.load(f, allow_pickle=True))    
                                        
                            except:
                                continue
                                
                        try:
                            
                            # if Top_three:
                            #     temp = []
                            #     RL_temp = []
                            #     for k in range(10):
                            #         try:
                            #             temp.append([RL[k][-1], k])
                            #         except:
                            #             continue
                                    
                            #     temp.sort()
                                    
                            #     RL_temp.append(RL[temp[-1][1]])
                            #     RL_temp.append(RL[temp[-2][1]])
                            #     RL_temp.append(RL[temp[-3][1]])
                                
                            #     RL = RL_temp
                            
                            mean = np.mean(np.array(RL), 0)
                            steps = np.linspace(0, 100*4096, len(mean))
                            std = np.std(np.array(RL),0)
                            axes.plot(steps, mean, label=policy_label, c=colors[policy])
                            axes.fill_between(steps, mean-std, mean+std, alpha=0.2, facecolor=colors[policy])
                            
                            axes.set_ylim([0, 1.05])
                            # axes.set_xlim([0, 200000])
                            axes.set_xlabel('Frames')
                            axes.set_ylabel('Reward')
                            
                            print(f'dataset: {data}, agent: {policy}, reward: {mean[-1]}, std: {std[-1]} , DA: {domain_adapt}, ri: {ri}')
                            charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: {ri}']['mean'] = mean[-1]
                            charts[f'dataset: {data}, agent: {policy}, DA: {domain_adapt}, ri: {ri}']['std'] = std[-1]
                            
                        except:
                            continue
        
            if not os.path.exists(f"./Figures/{env}"):
                os.makedirs(f"./Figures/{env}")
                      
            handles, labels = axes.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=2, prop={'size': 12})
            plt.title(f'{env}+Rodent, AWPO w/ Off-Obs, ri = {ri}')  
                   
            plt.savefig(f'Figures/{env}/on_off_RL_from_observations_{env}_dataset_{data}_intrinsic_reward_{ri}.pdf', format='pdf', bbox_inches='tight')

# %%
labels = ['AWPO+GAE', 'AWPO+GAE, \n $D_S: Fig.1b$, \n DA: False, \n ri: 0.01', 'AWPO+GAE, \n $D_S: Fig.1b$, \n DA: True, \n ri: 0.01', 'AWPO+PQL', 'AWPO+PQL, \n $D_S: Fig.1b$, \n DA: False, \n ri: 0.01', 'AWPO+PQL, \n $D_S: Fig.1b$, \n DA: True, \n ri: 0.01']
x_pos = np.arange(len(labels))
means = [0.2, 0.5, 0.22, 0.02, 0.06, 0.003]
std = [0.32/np.sqrt(10), 0.26/np.sqrt(10), 0.27/np.sqrt(10), 0.03/np.sqrt(10), 0.13/np.sqrt(10), 0.009/np.sqrt(10)]

# Build the plot
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x_pos, means, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Reward', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Figures/D_S_human.pdf', format='pdf', bbox_inches='tight')
plt.show()
            
            
# %%

labels = ['AWPO+GAE', 'AWPO+GAE, \n $D_S: Fig.1c$, \n DA: False, \n ri: 0.01', 'AWPO+GAE, \n $D_S: Fig.1c$, \n DA: True, \n ri: 0.01', 'AWPO+PQL', 'AWPO+PQL, \n $D_S: Fig.1c$, \n DA: False, \n ri: 0.01', 'AWPO+PQL, \n $D_S: Fig.1c$, \n DA: True, \n ri: 0.01']
x_pos = np.arange(len(labels))
means = [0.2, 0.3, 0.4, 0.02, 0.04, 0.06]
std = [0.32/np.sqrt(10), 0.3/np.sqrt(10), 0.35/np.sqrt(10), 0.03/np.sqrt(10), 0.08/np.sqrt(10), 0.13/np.sqrt(10)]

# Build the plot
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x_pos, means, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Reward', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Figures/D_S_rodent.pdf', format='pdf', bbox_inches='tight')
plt.show()            

# %%

labels = ['AWPO+GAE', 'AWPO+GAE, \n $D_S: 1(c)$, \n DA: True, \n ri: 0.005', 'AWPO+PQL', 'AWPO+PQL, \n $D_S: 1(c)$, \n DA: True, \n ri: 0.005']
x_pos = np.arange(len(labels))
means = [0.2, 0.45, 0.02, 0.03]
std = [0.32, 0.25, 0.03, 0.07]

# Build the plot
fig, ax = plt.subplots(figsize=(10,5))
ax.bar(x_pos, means, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Reward', fontsize=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
plt.savefig('Figures/D_S_rodent_r_i_0.005.pdf', format='pdf', bbox_inches='tight')
plt.show()       
    
    
