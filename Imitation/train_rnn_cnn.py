import numpy as np
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#import torch.nn.functional as F
#import torch.optim as optim
#from torch.distributions import Categorical
#from torch.autograd import Variable
import csv
import json
from sklearn.model_selection import train_test_split

S_statespace = 3
S_actionspace = 6


#!!!!!! https://discuss.pytorch.org/t/lstm-how-to-remember-hidden-and-cell-states-across-different-batches/11957/4
# on use, our "batches" are 1, so we need to save them and provide them on each step

def print_board(flattened_board):
    for i in range(11):
        row = flattened_board[i*11:(i+1)*11]
        print(row)

def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)

# as we dont know the reward yet, we only train the actor, not the critic
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(S_statespace, 64, 3, stride=1, padding=1)
        #self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        #self.maxp4 = nn.MaxPool2d(2, 2)

        self.encoder1 = nn.Linear(14483,1000)
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
        hx,cx = self.actor_lstm(x,(hx,cx))
        action = self.actor_out(hx)
        return action,value, (hx, cx)

    def get_lstm_reset(self):
        hx = torch.zeros(1, 6) 
        cx = torch.zeros(1, 6)
        return hx,cx

def pred(X):
    X = get_variable(Variable(torch.from_numpy(X)))
    y = net(X)
    return y.data.numpy()

def onehot(t, num_classes):
    out = np.zeros((t.shape[0], num_classes))
    for row, col in enumerate(t):
        out[row, col] = 1
    return out

def accuracy(ys, ts):
    #expect: ys: onehot encoded, targets: index
    correct_prediction = torch.eq(torch.max(ys, 1)[1], ts)
    return torch.mean(correct_prediction.float())


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x

def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()
    return x.data.numpy()


if __name__ == '__main__':
    num_actions = 6
    batch_size = 5
    num_epochs = 5
    train_size = 100000000000
    model = Net()
    model.cuda()
    hn,cn = model.get_lstm_reset()
    hn.cuda()
    cn.cuda()
    
    #eps = np.finfo(np.float32).eps.item()

    states_obs = []
    states_raw = []
    actions = []
    logFile_actions = 'simpleAgentActions_sequence_rawObs.txt'
    logFile_states_obs = 'simpleAgentStates_obs.txt'
    logFile_states_raw = 'simpleAgentStates_raw.txt'
    with open(logFile_actions,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [int(w) for w in line]
                actions.append(line)

    with open(logFile_states_obs,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [json.loads(l) for l in line]
                line = [[np.asarray(l)] for l in line]
                states_obs.append(line)

    with open(logFile_states_raw,'r') as fp:
        re = csv.reader(fp, dialect='excel')
        for i,line in enumerate(re):
            if len(line)>0 and i<train_size:
                line = [json.loads(l) for l in line]
                states_raw.append(np.asarray(line))

    n = len(actions)
    actions = torch.from_numpy((np.array(actions,dtype=np.long)))
    states_obs = torch.from_numpy(np.asarray(states_obs,dtype=np.float32))
    states_raw = torch.from_numpy(np.asarray(states_raw,dtype=np.float32))
  
    nr_train = int(0.9*n//1)
    nr_val = int(n-nr_train)

    perm_ind = torch.randperm(n)
    train_ind = perm_ind[0:nr_train]
    val_ind = perm_ind[nr_train:]

    X1_tr = states_obs[train_ind]
    X2_tr = states_raw[train_ind]
    y_tr = actions[train_ind]
    X1_val = states_obs[val_ind]
    X2_val = states_raw[val_ind]
    y_val = actions[val_ind]


    print('train size {}'.format(nr_train))
    print('validation size {}'.format(nr_val))


    # store loss and accuracy for information
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    #tr_input = get_variable(torch.from_numpy(X_tr))
    #tr_targets = get_variable(torch.from_numpy(y_tr)).long()
    # training loop
    for e in range(num_epochs):
        # get training input and expected output as torch Variables and make sure type is correct
        
        permutation = torch.randperm(nr_train)
        for i in range(0,nr_train, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            tr_input1_batch, tr_input2_batch, tr_targets_batch = get_variable(X1_tr[indices]).squeeze(2), get_variable(X2_tr[indices]).squeeze(2), get_variable(y_tr[indices]).long()

            siz = tr_targets_batch.size()

            hn1_batch = hn.repeat(2,siz[0],1).to(torch.device("cuda"))
            cn1_batch = cn.repeat(2,siz[0],1).to(torch.device("cuda"))
            
            # zeroize accumulated gradients in parameters
            optimizer.zero_grad()
            tr_output,_,_ = model(tr_input1_batch,tr_input2_batch,hn1_batch,cn1_batch)
            tr_output = tr_output.permute(0,2,1) #switch dimensions for criterion
            batch_loss =criterion(tr_output, tr_targets_batch)

            batch_loss.backward()
            optimizer.step()
            train_acc = accuracy(tr_output, tr_targets_batch)
            #print('{}\t{}\t{}'.format(batch_loss.data,torch.equal(a.data,b.data),(model.affine3.weight.data)[0]))
    
            # store training loss
            train_losses.append(get_numpy(batch_loss))
            train_accs.append(train_acc)
            del tr_input1_batch
            del tr_input2_batch
            del tr_targets_batch
            del tr_output
            del batch_loss
            del hn1_batch
            del cn1_batch
            del train_acc
            model.zero_grad()
    
        #dump_tensors()
        torch.cuda.empty_cache()
        model.eval()
        va = []
        bl = 0
        permutation = torch.randperm(nr_val)
        for i in range(0,nr_val, batch_size):
            indices = permutation[i:i+batch_size]
            val_input1_batch, val_input2_batch, val_targets_batch = get_variable(X1_val[indices]).squeeze(2), get_variable(X2_val[indices]).squeeze(2), get_variable(y_val[indices]).long()
            # get validation input and expected output as torch Variables and make sure type is correct

            siz = val_targets_batch.size()
            hn1_batch = hn.repeat(2,siz[0],1).to(torch.device("cuda"))
            cn1_batch = cn.repeat(2,siz[0],1).to(torch.device("cuda")) 

            # predict with validation input
            val_output,_,_ = model(val_input1_batch,val_input2_batch,hn1_batch,cn1_batch)
            val_output = val_output.permute(0,2,1) #switch dimensions for criterion
            val_acc = accuracy(val_output, val_targets_batch)
            batch_loss =criterion(val_output, val_targets_batch)
            va.append(val_acc.item())
            bl += get_numpy(batch_loss)
            del val_input1_batch
            del val_input2_batch
            del val_targets_batch
            del val_output
            del batch_loss
            del hn1_batch
            del cn1_batch
            del val_acc
            torch.cuda.empty_cache()
    
        model.train()
    
        # store loss and accuracy
        val_losses.append(bl)
        val_accs.append(np.mean(va))
    
        #if e % 10 == 0:
        print("Epoch %i, "
                "Train Cost: %0.3f"
                "\tVal Cost: %0.3f"
                "\t Val acc: %0.3f" % (e, 
                                        train_losses[-1],
                                        val_losses[-1],
                                        val_accs[-1]))

    torch.save(model.state_dict(), './4xCNNx_2xRNN.pth')

    plt.figure()
    epoch_1 = np.linspace(0,len(val_losses)-1,len(train_losses))
    epoch_2 = np.arange(len(val_losses))
    plt.plot(epoch_1, train_losses, 'r', label='Train Loss')
    plt.plot(epoch_2, val_losses, 'b', label='Val Loss')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.plot(epoch_1, train_accs, 'r', label='Train Acc')
    plt.plot(epoch_2, val_accs, 'b', label='Val Acc')
    plt.legend()
    plt.xlabel('Updates')
    plt.ylabel('Accuracy')
    plt.show()