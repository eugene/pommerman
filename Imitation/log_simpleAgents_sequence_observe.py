"""
Based on train_with_tensorforce example
With pytorch inspiration from https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
"""
import atexit
import functools
import os

import docker

from .. import make
import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from .. import agents
import csv
import json

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]



def observe(state,action_history):
    obs_width = 5 #choose uneven number
    obs_radius = obs_width//2
    board = state['board']
    blast_strength = state['bomb_blast_strength']
    bomb_life = state['bomb_life']
    pos = np.asarray(state['position'])
    board_pad = np.pad(board,(obs_radius,obs_radius),'constant',constant_values=1)
    BS_pad = np.pad(blast_strength,(obs_radius,obs_radius),'constant',constant_values=0)
    life_pad = np.pad(bomb_life,(obs_radius,obs_radius),'constant',constant_values=0)
    #centered, padded board
    board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    bomb_BS_cent = BS_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    bomb_life_cent = life_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]
    ammo = np.asarray([state['ammo']])
    my_BS = np.asarray([state['blast_strength']])

    #note: on the board, 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
    out = np.empty((3,11+2*obs_radius,11+2*obs_radius),dtype=np.float32)
    out[0,:,:] = board_pad
    out[1,:,:] = BS_pad
    out[2,:,:] = life_pad
    #get raw surroundings
    raw = np.concatenate((board_cent.flatten(),bomb_BS_cent.flatten()),0)
    raw = np.concatenate((raw,bomb_life_cent.flatten()),0)
    raw = np.concatenate((raw,ammo),0)
    raw = np.concatenate((raw,my_BS),0)
    raw = np.concatenate((raw,action_history),0)

    return out,raw


def main():
    config = "PommeFFACompetition-v0"
    game_state_file = None

    myAgents = [agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
    

    env = make(config, myAgents, game_state_file)


    logFile_states_raw = 'simpleAgentStates_raw.txt'
    logFile_states_obs = 'simpleAgentStates_obs.txt'
    logFile_actions = 'simpleAgentActions_sequence_rawObs.txt'
    for i_episode in range(5000):
        #render every 50'th episode
        #args.render = not(i_episode % 50)
        state = env.reset()
        k = list(state[0].keys())
        raw_states = []
        obs_states = []
        SA_actions = []
        action_history = np.zeros(6)
        for t in range(10000):  # Don't infinite loop while learning
            agent_actions = env.act(state)
            for i in range(1): #try to only log one agent
                #we make a list from position, board, bomb blast strength, bomb life, blast strength, can kick and ammo
                #if agent is alive
                if 10 in state[i][k[0]]:
                    obs,raw = observe(state[i],action_history)
                    raw_states.append(raw.tolist())
                    obs_states.append(obs.tolist())
                    SA_actions.append(agent_actions[i])

                    action_history[:-1] = action_history[1:]
                    action_history[-1] = agent_actions[i]
            state, reward, done, _ = env.step(agent_actions)
            if t==100 and 10 in state[0][k[0]]:
                with open(logFile_states_obs,'a') as fp:
#                    obs_states = [[int(o) for o in inner_list] for inner_list in obs_states]
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(obs_states)
                with open(logFile_states_raw,'a') as fp:
                    #raw_states = [[int(o) for o in inner_list] for inner_list in raw_states]
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(raw_states)
                with open(logFile_actions,'a') as fp:
                    wr = csv.writer(fp, dialect='excel')
                    wr.writerow(SA_actions)
                print(i_episode)
                break
            if done or not(10 in state[0][k[0]]):
                print(i_episode)
                break


if __name__ == "__main__":
    main()
