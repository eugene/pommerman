import pommerman
from pommerman import agents
import sys
import gym
import time
import random
import numpy as np
from collections import namedtuple
from collections import Counter
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# import matplotlib
# import matplotlib.pyplot as plt
# plt.ion()

class Leif(agents.BaseAgent):
    def __init__(self, model):
        super(Leif, self).__init__()
        self.model     = model
        self.states    = []
        self.actions   = []
        self.hidden    = []
        self.values    = []
        self.probs     = []
        self.debug     = False
        self.stochastic = True
        
    def translate_obs(self, o):
        obs_width = self.model.obs_width
        
        board = o['board'].copy()
        agents = np.column_stack(np.where(board > 10))

        for i, agent in enumerate(agents): 
            agent_id = board[agent[0], agent[1]]
            if agent_id not in o['alive']: # < this fixes a bug >
                board[agent[0], agent[1]] = 0
            else:
                board[agent[0], agent[1]] = 11

        obs_radius = obs_width//2
        pos = np.asarray(o['position'])

        # board
        board_pad = np.pad(board, (obs_radius,obs_radius), 'constant', constant_values=1)
        self.board_cent = board_cent = board_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb blast strength
        bbs = o['bomb_blast_strength']
        bbs_pad = np.pad(bbs, (obs_radius,obs_radius), 'constant', constant_values=0)
        self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        # bomb life
        bl = o['bomb_life']
        bl_pad = np.pad(bl, (obs_radius,obs_radius), 'constant', constant_values=0)
        self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0]+2*obs_radius+1,pos[1]:pos[1]+2*obs_radius+1]

        return np.concatenate((
            board_cent, bbs_cent, bl_cent,
            o['blast_strength'], o['can_kick'], o['ammo']), axis=None)

    def act(self, obs, action_space):
        obs = self.translate_obs(obs)
        
        last_hn, last_cn = self.hidden[-1][0], self.hidden[-1][1]
        obs = torch.from_numpy(obs).float().to(self.model.device)
        
        with torch.no_grad():
            self.model.eval()
            last_hn, last_cn = torch.tensor(last_hn).unsqueeze(0), torch.tensor(last_cn).unsqueeze(0)
            probs, val, hn, cn = self.model(obs.unsqueeze(0), last_hn, last_cn, self.debug)
            
            if self.debug: 
                print("hn mean:", hn.mean(), "hn std:", hn.std(), "cn mean:", cn.mean(), "cn std:", cn.std())
            
            probs_softmaxed = F.softmax(probs, dim=-1)

            if self.stochastic: 
                action = Categorical(probs_softmaxed).sample().item()
            else: 
                action = probs_softmaxed.max(1, keepdim=True)[1].item()

        self.actions.append(action)
        self.states.append(obs.squeeze(0).numpy())
        self.probs.append(probs.detach().numpy())
        self.values.append(val.detach().item())
        self.hidden.append(
            ( hn.squeeze(0).clone().detach().numpy(), 
              cn.squeeze(0).clone().detach().numpy() ))

        return action
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.hidden[:]
        del self.probs[:]
        del self.values[:]

        self.hidden.insert(0, self.model.init_rnn())

        return self.states, self.actions, self.hidden, self.probs, self.values

class Stoner(agents.BaseAgent):
        def __init__(self): super(Stoner, self).__init__()
        def act(self, obs, action_space): 
            return 0#random.randint(1,4) #0

class A2CNet(nn.Module):
    def __init__(self, gpu = True): 
        super(A2CNet, self).__init__()
        
        self.gamma             = 0.50   # Discount factor for rewards (default 0.99)
        self.entropy_coef      = 0.01   # Entropy coefficient (0.01)
        self.obs_width = w     = 17     # Window width/height (must be uneven)
        self.lr                = 0.001  # 3e-2

        self.inputs_to_conv = ic  = 3*(w**2)           # 3 boards
        self.inputs_to_fc   = ifc = 3                  # blast strength, can_kick, ammo
        self.conv_channels  = cc  = 45                 # number of conv outputs
        #self.flat_after_c  = fac = cc * (w-3) * (w-3) # cc * (w-4) * (w-4) # flattened num after conv
        #self.flat_after_c  = fac = cc * (w-2) * (w-2) # cc * (w-4) * (w-4) # flattened num after conv
        #self.flat_after_c  = fac = cc * (w-cc) * (w-cc) # cc * (w-4) * (w-4) # flattened num after conv
        self.flat_after_c  = fac = 13005

        self.fc1s, self.fc2s, self.fc3s = 1024, 512, 64
        
        self.rnn_input_size   = self.fc2s 
        self.rnn_hidden_size  = 64
        
        self.conv1 = torch.nn.Conv2d(3,  cc,   kernel_size=3, stride=1, padding=1, groups=3)
        self.conv2 = torch.nn.Conv2d(cc, cc,   kernel_size=3, stride=1, padding=1, groups=3) #dilation=2, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(cc, cc,   kernel_size=3, stride=1, padding=1, groups=3)
        self.conv4 = torch.nn.Conv2d(cc, cc,   kernel_size=3, stride=1, padding=1, groups=3)

        self.bn1, self.bn2 = nn.BatchNorm2d(cc), nn.BatchNorm2d(cc)
        self.bn3, self.bn4 = nn.BatchNorm2d(cc), nn.BatchNorm2d(cc)

        self.fc_after_conv1 = nn.Linear(fac, self.fc1s)
        self.fc_after_conv2 = nn.Linear(self.fc1s + ifc, self.fc2s)
        self.fc_after_conv3 = nn.Linear(self.fc2s, self.fc2s)
        self.fc_after_conv4 = nn.Linear(self.fc2s, self.fc2s)
        
        self.rnn = torch.nn.LSTMCell(self.rnn_input_size, self.rnn_hidden_size)

        self.fc_after_rnn_1 = nn.Linear(self.rnn_hidden_size, self.fc3s)
        # self.fc_after_rnn_2 = nn.Linear(self.fc3s, self.fc3s)
        # self.fc_after_rnn_3 = nn.Linear(self.fc3s, self.fc3s)
        # self.fc_after_rnn_4 = nn.Linear(self.fc3s, self.fc3s)
        
        self.action_head = nn.Linear(self.fc3s, 6)
        self.value_head  = nn.Linear(self.fc2s, 1)

        self.optimizer   = optim.Adam(self.parameters(), lr=self.lr)
        self.eps         = np.finfo(np.float32).eps.item()
        
        self.device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")
        if self.device.type == 'cuda': self.cuda()
        return None

    def forward(self, x, hn, cn, debug = False):
        batch_size = x.shape[0]
        w, wh = self.obs_width, self.obs_width**2
        
        boards  = x[:,      0:wh].view(batch_size, 1, w, w)
        bbs     = x[:,   wh:wh*2].view(batch_size, 1, w, w)
        bl      = x[:, wh*2:wh*3].view(batch_size, 1, w, w)
        
        rest    = x[:, wh*3:]
        to_conv = torch.cat([boards, bbs, bl], 1)
        
        xc = self.conv1(to_conv)
        xc = self.bn1(xc)
        xc = F.relu(xc)

        xc = self.conv2(xc)
        xc = self.bn2(xc)
        xc = F.relu(xc)

        xc = self.conv3(xc)
        xc = self.bn3(xc)
        xc = F.relu(xc)

        xc = self.conv4(xc)
        xc = self.bn4(xc)
        xc = F.relu(xc)
        
        xc = xc.view(batch_size, -1)
        xc = self.fc_after_conv1(xc)
        xc = F.relu(xc)
        
        xc = torch.cat((xc, rest), 1)
        xc = self.fc_after_conv2(xc)
        xc = F.relu(xc)

        xc = self.fc_after_conv3(xc)
        xc = F.relu(xc)
        
        xc = self.fc_after_conv4(xc)
        xc = F.relu(xc)
        
        # if not debug:
        #     print(xc[0, :].mean(), xc[0, :].std())

        if debug == True:   
            mm = xc[0, :].mean()
            nn = xc[0, :].std()
        
        values  = self.value_head(xc)
        hn, cn  = self.rnn(xc, (hn, cn))
        xc = hn #torch.cat((xc, hn), 1)
        
        if debug == True:
            mm1 = xc[0, :].mean()
            nn1 = xc[0, :].std()
            print("Before rnn:", (mm,nn), "After rnn:", (mm1,nn1))
        
        xc = self.fc_after_rnn_1(xc)
        xc = F.relu(xc)
        
        # xc = self.fc_after_rnn_2(xc)
        # xc = F.relu(xc)

        # xc = self.fc_after_rnn_3(xc)
        # xc = F.relu(xc)
        
        # xc = self.fc_after_rnn_4(xc)
        # xc = F.relu(xc)
        
        probs = self.action_head(xc)
        
        return probs, values, hn, cn
        
    def init_rnn(self):
        device = self.device
        s = self.rnn_hidden_size
        return (torch.zeros(s).detach().numpy(), torch.zeros(s).detach().numpy())
    
    def discount_rewards(self, _rewards):
        R = 0
        gamma = self.gamma
        rewards = []
        for r in _rewards[::-1]:
            R = r + gamma * R
            rewards.insert(0, R)

        # rewards = np.array(rewards) 
        # rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)
        
        return rewards #torch.from_numpy(rewards).to(self.device)

def naked_env(agent_list):
    env = gym.make('PommeFFACompetition-v0')
    env._num_items = 0
    env._num_wood  = 0
    env._num_rigid = 0
    env._max_steps = 100

    for id, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        agent.init_agent(id, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(None)
    env.set_render_mode('human')
    return env

def normal_env(agent_list):
    env = gym.make('PommeFFACompetition-v0')
    
    for id, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        agent.init_agent(id, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(None)
    env.set_render_mode('human')
    return env