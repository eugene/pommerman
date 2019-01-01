from threading import Thread
from model import *
import pommerman
import colorama
from pommerman import agents
from collections import Counter
import os
import sys
import time
import random
import os
import math
#import pickle
import os

ROLLOUTS_PER_BATCH = 1
batch = []

class World():
    def __init__(self, init_gmodel = True):
        if init_gmodel: 
            self.gmodel = A2CNet(gpu = True) # Global model

        self.model = A2CNet(gpu = False)     # Agent (local) model
        self.leif = Leif(self.model)
        self.stoner = Stoner()

        self.agent_list = [
            self.leif, 
            #self.stoner
            agents.SimpleAgent(), 
            agents.SimpleAgent(), 
            agents.SimpleAgent()
        ]
        self.env = normal_env(self.agent_list) #naked_env
        fmt = {
            'int':   self.color_sign,
            'float': self.color_sign
        }
        np.set_printoptions(formatter=fmt,linewidth=300)
        pass

    def color_sign(self, x):
        if x == 0:    c = colorama.Fore.LIGHTBLACK_EX
        elif x == 1:  c = colorama.Fore.BLACK
        elif x == 2:  c = colorama.Fore.BLUE
        elif x == 3:  c = colorama.Fore.RED
        elif x == 4:  c = colorama.Fore.RED
        elif x == 10: c = colorama.Fore.YELLOW
        else:         c = colorama.Fore.WHITE
        x = '{0: <2}'.format(x)
        return f'{c}{x}{colorama.Fore.RESET}'

def do_rollout(env, leif, do_print = False):
    done, state = False, env.reset()
    rewards, dones   = [], []
    states, actions, hidden, probs, values = leif.clear()

    while not done:
        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])
            
        action = env.act(state)
        state, reward, done, info = env.step(action)
        if reward[0] == -1: done = True
        rewards.append(reward[0])
        dones.append(done)
    
    hidden = hidden[:-1].copy()
    hns, cns = [], []
    for hns_cns_tuple in hidden:
        hns.append(hns_cns_tuple[0])
        cns.append(hns_cns_tuple[1])
    
    return (states.copy(), 
            actions.copy(), 
            rewards, dones, 
            (hns, cns), 
            probs.copy(), 
            values.copy())

def gmodel_train(gmodel, states, hns, cns, actions, rewards, gae):
    states, hns, cns = torch.stack(states), torch.stack(hns, dim=0), torch.stack(cns, dim=0)
    gmodel.train()
    probs, values, _, _ = gmodel(states.to(gmodel.device), hns.to(gmodel.device), cns.to(gmodel.device), debug=False)
    
    prob      = F.softmax(probs, dim=-1)
    log_prob  = F.log_softmax(probs, dim=-1)
    entropy   = -(log_prob * prob).sum(1)

    log_probs = log_prob[range(0,len(actions)), actions]
    advantages = torch.tensor(rewards).to(gmodel.device) - values.squeeze(1)
    value_loss  = advantages.pow(2)*0.5
    policy_loss = -log_probs*torch.tensor(gae).to(gmodel.device) - gmodel.entropy_coef*entropy
    
    gmodel.optimizer.zero_grad()
    pl = policy_loss.sum()
    vl = value_loss.sum()
    loss = pl+vl
    loss.backward()
    gmodel.optimizer.step()
    
    return loss.item(), pl.item(), vl.item()

def unroll_rollouts(gmodel, list_of_full_rollouts):
    gamma = gmodel.gamma
    tau   = 1

    states, actions, rewards, hns, cns, gae = [], [], [], [], [], []
    for (s, a, r, d, h, p, v) in list_of_full_rollouts:
        states.extend(torch.tensor(s))
        actions.extend(a)
        rewards.extend(gmodel.discount_rewards(r))
        
        hns.extend([torch.tensor(hh) for hh in h[0]])
        cns.extend([torch.tensor(hh) for hh in h[1]])

        # Calculate GAE
        last_i, _gae, __gae = len(r) - 1, [], 0
        for i in reversed(range(len(r))):
            next_val = v[i+1] if i != last_i else 0
            delta_t = r[i] + gamma*next_val - v[i]
            __gae = __gae * gamma * tau + delta_t
            _gae.insert(0, __gae)

        gae.extend(_gae)
    
    return states, hns, cns, actions, rewards, gae

def train(world):
    model, gmodel = world.model, world.gmodel
    leif, env     = world.leif, world.env
    
    if False and os.path.isfile("convrnn-s.weights"):
        model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
        gmodel.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))
    
    if os.path.exists("training.txt"): os.remove("training.txt")

    rr = -1
    ii = 0
    for i in range(40000):
        full_rollouts = [do_rollout(env, leif) for _ in range(ROLLOUTS_PER_BATCH)]
        last_rewards = [roll[2][-1] for roll in full_rollouts]
        not_discounted_rewards = [roll[2] for roll in full_rollouts]
        states, hns, cns, actions, rewards, gae = unroll_rollouts(gmodel, full_rollouts)
        gmodel.gamma = 0.5+1/2./(1+math.exp(-0.0003*(i-20000))) # adaptive gamma
        l, pl, vl = gmodel_train(gmodel, states, hns, cns, actions, rewards, gae)
        rr = rr * 0.99 + np.mean(last_rewards)/ROLLOUTS_PER_BATCH * 0.01
        ii+=len(actions)
        print(i, "\t", round(gmodel.gamma, 3), round(rr,3), "\twins:", last_rewards.count(1), Counter(actions), round(sum(rewards),3), round(l,3), round(pl,3), round(vl,3))
        with open("training.txt", "a") as f: print(rr, "\t", round(gmodel.gamma,4), "\t", round(vl,3),"\t", round(pl,3),"\t", round(l,3), file=f)
        model.load_state_dict(gmodel.state_dict())
        if i >= 10 and i % 300 == 0: torch.save(gmodel.state_dict(), "convrnn-s.weights") 

def run(world):
    done, ded, state, _ = False, False, world.env.reset(), world.leif.clear()
    
    while not done:
        action = world.env.act(state)
        state, reward, done, info = world.env.step(action)
        print(world.leif.board_cent)
        print(world.leif.bbs_cent)
        print(world.leif.bl_cent)
        time.sleep(0.2)

    world.env.close()    
    return None

def eval(world, init_gmodel = False):
    env = world.env
    model = world.model
    leif = world.leif
    leif.debug = True
    leif.stochastic = False

    do_print = True
    done = None
    reward = 0
    last_reward = [0,0,0,0]
    
    while True:
        model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))

        done, state, _ = False, env.reset(), leif.clear()
        t = 0
        while not done:
            if do_print:
                time.sleep(0.1)
                os.system('clear')
                print(state[0]['board'])
                print("\n\n")
                print("Probs: \t", leif.probs[-1] if len(leif.probs) > 0 else [])
                print("Val: \t", leif.values[-1] if len(leif.values) > 0 else None)
                print("\nLast reward: ", last_reward, "Time", t)

            action = env.act(state)
            state, reward, done, info = env.step(action)
            if reward[0] == -1: 
                last_reward = reward
                break
            t+=1

def readme(world):
    print("Usage: ")
    print("\t to train:\tpython main.py train")
    print("\t to evaluate:\tpython main.py eval\n\n")
    print("Procedure:")
    print("Start the training. Wait for 300 episodes (this will generate weights file). Run evaluate. See running results.")
    return None

entrypoint = next(iter(sys.argv[1:]), "readme")
locals()[entrypoint](World())