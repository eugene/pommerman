from sharedAdam import *
#on windows, multiprocessing: https://pytorch.org/docs/stable/notes/windows.html
#use one thread for parallel as they will block each other otherwise (https://github.com/ikostrikov/pytorch-a3c/issues/33)
os.environ["OMP_NUM_THREADS"] = "1"

#define globals
S_statespace = 3
S_actionspace = 6

UPDATE_GLOBAL_ITER = 800
GAMMA = 0.95
LAMBDA = 1
MAX_EP = 2000
eps = np.finfo(np.float32).eps.item()

def load_checkpoint(filename,model,optimizer):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'".format(filename))
    else:
        print("=> no checkpoint found at '{}'".format(filename))
    return model,optimizer

def save_checkpoint(filename,model,optimizer):
    state = {
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }
    torch.save(state, filename)

def test(gnet):
    John = A3CAgent(gnet)
    John.set_train(False)
    agentList = [John, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
    env = pommerman.make('PommeFFACompetition-v0', agentList)
    wins = []
    for ii in range(100):
        John.reset_lstm()
        state = env.reset()
        done = False
        while done == False:
            if ii%20 == 0:
                env.render()
            # time.sleep(1/24)
            with torch.no_grad(): actions = env.act(state)
            state_next, reward, done, info = env.step(actions)

        print(ii, "DONE. Info:", info, "reward:", reward, "You win = ", info['winners'][0] == 0 if info['result'].name == 'Win' else False)
        wins.append(info['winners'][0] if info['result'].name == 'Win' else -1)
    
    fig, ax = plt.subplots(num=1, clear=True)
    winrate = wins.count(0) / len(wins)
    fig, ax = plt.subplots(num=1, clear=True)
    t, p0, p1, p2, p3 = plt.bar([-1, 0, 1, 2, 3], [
            wins.count(-1) / len(wins) * 100, 
            wins.count(0)  / len(wins) * 100, 
            wins.count(1)  / len(wins) * 100, 
            wins.count(2)  / len(wins) * 100, 
            wins.count(3)  / len(wins) * 100])
    t.set_facecolor('b')

    p0.set_facecolor('r')
    p1.set_facecolor('g')
    p2.set_facecolor('b')
    p3.set_facecolor('c')
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xticklabels(['Ties', 'Agent\n(A2C)', 'Agent 1\nSimpleAgent', 'Agent 2\nSimpleAgent', 'Agent 3\nSimpleAgent'])
    ax.set_ylim([0, 100])
    ax.set_ylabel('Percent')
    ax.set_title('Bomberman. FFA mode.')
    print("Winrate: ", winrate)
    plt.show()

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]

def ensure_shared_grads(lnet, gnet):
    for param, shared_param in zip(lnet.parameters(),gnet.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def update_glob_net(opt, lnet, gnet, agent, GAMMA):
    R = 0
    policyLoss = 0
    valueLoss = 0
    gae = 0
    agent.values.append(torch.zeros(1)) # we need to add this for the deltaT equation
    #print(agent.rewards)
    for i in reversed(range(len(agent.rewards))):
        R = GAMMA * R + agent.rewards[i]
        advantage = R - agent.values[i]
        valueLoss = valueLoss + 0.5 * advantage.pow(2)
        deltaT = agent.rewards[i] + GAMMA * agent.values[i + 1].data - agent.values[i].data
        gae = gae * GAMMA * LAMBDA + deltaT
        policyLoss = policyLoss - agent.logProbs[i] * gae - 0.01 * agent.entropies[i]
    loss = (policyLoss + 0.5 * valueLoss)
    opt.zero_grad()
    loss.backward(retain_graph=True)
    ensure_shared_grads(lnet,gnet)
    opt.step()
    lnet.load_state_dict(gnet.state_dict())
    lnet.zero_grad()
    agent.clear_actions()

def record(global_ep, global_ep_r, ep_r, res_queue,global_nr_steps,nr_steps, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = -1#ep_r
            global_nr_steps.value = nr_steps
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
            global_nr_steps.value = global_nr_steps.value * 0.99 + nr_steps * 0.01
    res_queue.put(ep_r)
    print(
        name,
        "Ep:", global_ep.value,
        "| Avg Ep_r: %.2f" % global_ep_r.value,
        "| Avg Steps: %d" % global_nr_steps.value,
        "| Ep_r / Steps: %.2f" % (ep_r/nr_steps),
        #"| Ep_r: %.2f" % ep_r,
    )

def get_reward(state,old_state,agent_nr,start_reward,max_ammo,old_max_ammo,action,last_action,action_history_oh):
    #developer note: on the board, 0: nothing, 1: unbreakable wall, 2: wall, 3: bomb, 4: flames, 6,7,8: pick-ups:  11,12 and 13: enemies
    reward = 0
    #reward stage 0: teach the agent to move and make invalid actions (move into walls, place bombs when you have no ammo)
    ammo = old_state[agent_nr]['ammo']
    if action != 5:
        if state[agent_nr]['position']==old_state[agent_nr]['position']:
            reward-=0.03
    elif ammo==0:
        reward-=0.03

    #reward stage 1: teach agent to bomb walls (and enemies)
    #compute adjacent squares
    position = state[agent_nr]['position']
    adj = [(i,j) for i in (-1,0,1) for j in (-1,0,1) if not ((i == j) or i+j==0)]
    adjacent = numpy.matlib.repmat(position,4,1)
    adjacent = adjacent - np.asarray(adj)
    #limit adjacent squares to only include inside board
    adjacent = np.clip(adjacent,0,10)
    if action==5 and ammo>0:
        board = state[agent_nr]['board']
        for xy in adjacent:
            square_val = board[xy[0]][xy[1]]
            if square_val == 2:
                reward += 0.02
            elif square_val == 11 or square_val == 12 or square_val == 13:
                reward += 0.05

    ##reward stage2: teach agent to not stand on or beside bombs
    ##reward /= 4
    #bomb_life = state[agent_nr]['bomb_life']
    ##if we stand on a bomb or next to bomb
    #just_placed_bomb = np.logical_xor(last_action==5,action==5)
    #if bomb_life[position]>0 and not(just_placed_bomb):
    #    reward-=0.1*(9-bomb_life[position])
    #for xy in adjacent:
    #    if bomb_life[xy[0]][xy[1]]>0:
    #        reward -=0.05*(9-bomb_life[xy[0]][xy[1]])

    #reward agent for picking up power-ups
    blast_strength = state[agent_nr]['blast_strength']
    old_blast_strength = old_state[agent_nr]['blast_strength']
    can_kick = int(state[agent_nr]['can_kick'])
    old_can_kick = int(old_state[agent_nr]['can_kick'])
    reward += (can_kick-old_can_kick)*0.02
    reward += (max_ammo-old_max_ammo)*0.02
    reward += (blast_strength-old_blast_strength)*0.02


    #discourage action repetition by cross correlation of previous actions
    #corr = signal.correlate2d(action_history_oh,action_history_oh)[5:-5][0]
    #corr.sort()
    #reward -= corr[-2]/100   #the second highest correlation (highest is found at 0-lag and is always 10), 

    #only reward gameplay at last stage
    reward += start_reward
    return reward

class A3CNet(nn.Module):
    def __init__(self):
        super(A3CNet, self).__init__()
        self.conv1 = nn.Conv2d(S_statespace, 66, 3, stride=1,groups=3)
        self.conv2 = nn.Conv2d(66, 66, 3, stride=1, padding=1,groups=3)
        self.conv3 = nn.Conv2d(66, 66, 3, stride=1, padding=1,groups=3)
        self.conv4 = nn.Conv2d(66, 66, 3, stride=1, padding=1,groups=3)

        self.encoder1 = nn.Linear(11237,1000)
        self.encoder2 = nn.Linear(1000,200)
        self.encoder3 = nn.Linear(200,50)
        
        self.critic_linear = nn.Linear(83, 1)
        self.actor_lstm = nn.LSTM(50, S_actionspace,2,batch_first=True)
        self.actor_out = nn.Linear(S_actionspace, S_actionspace)

        torch.nn.init.xavier_uniform_(self.encoder1.weight)
        torch.nn.init.xavier_uniform_(self.encoder2.weight)
        torch.nn.init.xavier_uniform_(self.encoder3.weight)
        torch.nn.init.xavier_uniform_(self.critic_linear.weight)
        #torch.nn.init.xavier_uniform_(self.actor_linear.weight)

    def forward(self, x,raw, hx,cx):
        timesteps, batch_size, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(timesteps,batch_size, -1)
        x = torch.cat((x,raw),-1)
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = F.relu(self.encoder3(x))#.permute(1, 0, 2)
        #critic
        value = self.critic_linear(raw)
        #actor
        x, (hx,cx) = self.actor_lstm(x,(hx,cx))
        action = self.actor_out(x)
        return action,value, (hx, cx)

    def get_lstm_reset(self):
        hx = torch.zeros(2, 1, 6) 
        cx = torch.zeros(2, 1, 6)
        return hx,cx

class A3CAgent(agents.BaseAgent):
    def __init__(self, model):
        super(A3CAgent, self).__init__()
        self.model = model
        self.hn, self.cn = self.model.get_lstm_reset()
        #self.hx2, self.cx2 = self.model.get_lstm_reset()
        self.rewards = []
        self.values = []
        self.logProbs = []
        self.entropies = []
        self.action_history = np.zeros(6)
        self.train=True

    def act(self,state,action_space):
        if self.train:
            obs,raw = self.observe(state,self.action_history)
            logit,value,(hn,cn) = self.model(torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0),torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0),self.hn,self.cn)
            logit,value = logit.squeeze(0), value.squeeze(0) #remove batch dimension
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1)
            self.entropies.append(entropy)
            try:
                #action = torch.argmax(logit,dim=-1).unsqueeze(0)                #JUST TEST!!!
                action = Categorical(prob).sample().unsqueeze(0)
            except:
                print('obs1: ',obs1.data)
                print('logit: ',logit.data)
                print('value: ',value.data)
            log_prob = log_prob.gather(1, action)
            self.values.append(value)
            self.logProbs.append(log_prob)
            a = action.item()
        else:
            obs,raw = self.observe(state,self.action_history)
            logit,value,(hn,cn) = self.model(torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0),torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0),self.hn,self.cn)
            logit = logit.squeeze(0) #remove batch dimension
            prob = F.softmax(logit, dim=-1)
            a = torch.argmax(logit,dim=-1).item()
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = a
        return a

    def set_train(self,input):
        self.train=input

    def add_reward(self,reward):
        self.reward = max(min(reward, 1), -1)
        self.rewards.append(self.reward)

    def observe(self,state,action_history):
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

    def reset_lstm(self):
        self.hx, self.cx = self.model.get_lstm_reset()
        self.action_history = np.zeros(6)

    def clear_actions(self):
        self.values = []
        self.logProbs = []
        self.rewards = []
        self.entropies = []
        return self

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, global_nr_steps, res_queue, name):
        super(Worker, self).__init__()
        self.agent_nr = 0
        self.name = 'w%i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = A3CNet()
        self.lnet.load_state_dict(gnet.state_dict())
        self.lnet.train()
        #disable gradients for all layers in the model
        self.lnet = disable_model_grads(self.lnet)
        #enable gradients on only critic
        for p in self.lnet.critic_linear.parameters():
            p.requires_grad = True
        self.results = []
        self.global_nr_steps = global_nr_steps
        self.empty_oh_action = np.zeros((6,1))
        self.saved_oh_actions = np.repeat(self.empty_oh_action,6,1)

    def run(self):
        #If we move this to "init", we get an error on recursion depth
        self.A3CAgent = A3CAgent(self.lnet)
        self.agentList = [self.A3CAgent, agents.SimpleAgent(), agents.RandomAgent(), agents.RandomAgent()]
        self.env = env = pommerman.make('PommeFFACompetition-v0', self.agentList)

        total_step = 1
        while self.g_ep.value < MAX_EP:
            #Step 2) worker interacts with environment
            s_act = self.env.reset()
            r1 = 0
            max_ammo,old_max_ammo = 1,1
            ep_r = 0.
            self.render = False#self.g_ep.value%20==0
            self.A3CAgent.reset_lstm()
            if self.name == 'w0':
                enc1 = abs(torch.sum(self.gnet.encoder1.weight.data).item())
                enc2 = abs(torch.sum(self.gnet.encoder2.weight.data).item())
                enc3 = abs(torch.sum(self.gnet.encoder3.weight.data).item())
                conv1 = abs(torch.sum(self.gnet.conv1.weight.data).item())
                conv2 = abs(torch.sum(self.gnet.conv2.weight.data).item())
                conv3 = abs(torch.sum(self.gnet.conv3.weight.data).item())
                conv4 = abs(torch.sum(self.gnet.conv4.weight.data).item())
                cl = abs(torch.sum(self.gnet.critic_linear.weight.data).item())
                alstm1 = abs(torch.sum(self.gnet.actor_lstm.weight_ih_l0.data).item())
                alstm2 = abs(torch.sum(self.gnet.actor_lstm.weight_hh_l0.data).item())
                aout = abs(torch.sum(self.gnet.actor_out.weight.data).item())
                f = open("AbsSummedWeights_onlyCritic_v2.txt", "a")
                f.write('{0:.5f} \t {1:.5f} \t {2:.5f} \t {3:.5f} \t {4:.5f} \t {5:.5f} \t {6:.5f} \t {7:.5f} \t {8:.5f} \t {9:.5f} \t {10:.5f} \n'.format(enc1,enc2,enc3,conv1,conv2,conv3,conv4,alstm1,alstm2,aout,cl))
                f.close()
            while True:
                # only render worker 0
                if self.name == 'w0' and self.render:
                    self.env.render()
                agent_actions = self.env.act(s_act)         

                a = agent_actions[self.agent_nr]
                self.saved_oh_actions[:,:-1]=self.saved_oh_actions[:,1:]    #time shift
                self.saved_oh_actions[:,-1] = self.empty_oh_action[:,0]     #erase last value
                self.saved_oh_actions[a,-1] = 1                             #insert new onehot

                s_new, r, done, _ = self.env.step(agent_actions)
                done = done or r[self.agent_nr]==-1
                max_ammo = max(max_ammo,s_act[self.agent_nr]['ammo'])
                #reward and buffer
                r = r[self.agent_nr]
                #if (10 in s_act[self.agent_nr]['alive']) and total_step!=1:
                #    r = get_reward(s_new,s_act,self.agent_nr,r,max_ammo,old_max_ammo,a,a_old,self.saved_oh_actions)
                ep_r += r
                self.A3CAgent.add_reward(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    update_glob_net(self.opt, self.lnet, self.gnet, self.A3CAgent, GAMMA)
                    if done:
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue,self.global_nr_steps,s_new[self.agent_nr]['step_count'], self.name)
                        break
                s_act = s_new
                old_max_ammo = max_ammo
                a_old = a
                total_step += 1
        self.res_queue.put(None)

def disable_model_grads(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    return model

def main():
    global_ep, global_ep_r,global_nr_steps, res_queue = mp.Value('d', 0), mp.Value('d', 0.), mp.Value('d', 0.), mp.Queue()
    gnet = A3CNet()        # global network
    gnet.train()
    filename = './A3C_v10_cnn_lstm_trained_critic.pth'
    imitation_model = './4xCNNx_2xRNN_2.pth'
    gnet.load_state_dict(torch.load(imitation_model)) ## only first run!!!

    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=0.001)      # global optimizer
    #load previous training checkpoint if exists
    #load_checkpoint(filename,gnet,opt)
    #for g in opt.param_groups:
    #    g['lr'] = 0.00001

     #parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, global_nr_steps, res_queue, i) for i in range(mp.cpu_count()//2)]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    save_checkpoint(filename,gnet,opt)

    with open('imitation_performance.txt', 'a') as f:
        for item in res:
            f.write("%s\n" % item)


    print('joining workers')
    [w.join() for w in workers]


if __name__ == '__main__':
    main()
