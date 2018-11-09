from __future__ import print_function
import argparse
from utils_sinkhorn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import os

#........................Input path...........................................
OnServer = False
if OnServer:
    path = ''
else:
    path = ''

#...................Training settings.........................................
parser = argparse.ArgumentParser(description='PyTorch flocking predictor')
parser.add_argument('--batchsize', type=int, default=10, metavar='BS',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test_batchsize', type=int, default=10, metavar='BStest',
                    help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=1001, metavar='NEP',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--npoints', type=float, default=750, metavar='NPTS',
                    help='Number of particles')
parser.add_argument('--d', type=float, default=2, metavar='D',
                    help='Input measure dimension')
parser.add_argument('--nsimu', type=float, default=500, metavar='NSIM',
                    help='Number of simulations')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='LI',
                    help='random seed (default: 1)')
parser.add_argument('--eps', type=int, default=0.01, metavar='EPS',
                    help='Sinkhorn regularization strength')
parser.add_argument('--maxiter', type=int, default=2, metavar='MAXITER',
                    help='Number of Sinkhorn iterations')
args = parser.parse_args()
use_cuda = not args.no_cuda and torch.cuda.is_available()

print('Using gpu: '+str(use_cuda))
print( args )

torch.manual_seed(args.seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


#.......................Construct synthetic dataset...........................

# initial position and velocity
def crandn(n_simulations,n):
    return torch.randn(n_simulations,n,2).type(dtype)

def crandna(n_simulations,n,u,v):
    return torch.cat((u*torch.randn(n_simulations,n,1).type(dtype),v*torch.randn(n_simulations,n,1).type(dtype)),2)
          
# positive decreasing function of pairwise distances
def psi(r):
    r0 = 0.7
    return 1/(1+(r/r0)**2)**0.6

# absolute distance matrix
def D(x):
    return torch.sqrt( (torch.abs(x.unsqueeze(3)-x.transpose(1,2).unsqueeze(1))**2).sum(2) )

# laplacian
def L(x):
    return 1/(x.size(1))*( psi(D(x)) - diag( psi(D(x)).sum(1) ) )

# Build dataset
def store_flocking(n_simulations,n,datatype,path):
    q=1000 # total number of frames
    tau=1.3 # step size
    K=0.2
    all_simu_0 = torch.FloatTensor(n_simulations,2*n).type(dtype)
    v_0 = torch.FloatTensor(n_simulations,2*n).type(dtype)
    all_simu_2 = torch.FloatTensor(n_simulations,2*n).type(dtype)
    h=8
    nuage1=crandn(n_simulations,int(n/2))
    nuage1[:,:,0]-=h/2
    nuage1[:,:,1]+=h/4
    nuage2=crandn(n_simulations,int(n/2))
    nuage2[:,:,0]+=h/2
    nuage2[:,:,1]-=h/4
    x0=torch.cat((nuage1,nuage2),1)
    theta=np.pi/2*torch.rand((n_simulations,1)).repeat(1,int(n/2))
    torch.zeros(n_simulations,int(n/2),2).type(dtype)
    nuage1[:,:,0]=0.1*torch.cos(theta)
    nuage1[:,:,1]=-0.1*torch.sin(theta)
    nuage2=-nuage1
    v0=torch.cat((nuage1,nuage2),1)
    x=x0
    v=v0
    for i in range(q):
        x=x+tau*v
        v=v+tau*K*torch.bmm(L(x),v)
        x1=x-torch.mean(x,1).unsqueeze(1)
        if i==0:
            all_simu_0 = x1.reshape(n_simulations,2*n)
            v_0 = v.reshape(n_simulations,2*n)
        elif i==q-1:
            all_simu_2 = x1.reshape(n_simulations,2*n)
    result = torch.cat((all_simu_0,v_0,all_simu_2),1)
    np.save('flocking-'+datatype+'-'+str(n_simulations)+'-'+str(n),result.data.cpu().numpy())

print('Computing flocking database...')
store_flocking(args.nsimu,args.npoints,'train',path)
print('Train set done!')
store_flocking(args.test_batchsize,args.npoints,'test',path)
print('Test set done!')
print('Training...') 

#........................Load dataset.........................................
class FlockingCloudDataset(data.Dataset):
    """ Cucker-Smale flocking dataset.
    """
    def __init__(self, d, root):
        self.d = d
        self.npoints = args.npoints
        self.root = root
        xy = np.load(self.root)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:2*self.d*self.npoints]) # first (measures,velocities)
        self.y_data = torch.from_numpy(xy[:,2*self.d*self.npoints:]) # true final measures
    def __getitem__(self,index):
        mu_0 = self.x_data[index,:self.d*self.npoints].unsqueeze(0).float()
        v_0 = self.x_data[index,self.d*self.npoints:].unsqueeze(0).float()
        mu_2 = self.y_data[index]
        return torch.cat((mu_0,v_0),1).squeeze(0), mu_2
    def __len__(self):
        return self.len

train_loader = data.DataLoader(dataset=FlockingCloudDataset(d=args.d,root='flocking-train-'+str(args.nsimu)+'-'+str(args.npoints)+'.npy'),
                               batch_size=args.batchsize,
                               shuffle=True,
                               **kwargs)

test_loader = data.DataLoader(dataset=FlockingCloudDataset(d=args.d,root='flocking-test-'+str(args.test_batchsize)+'-'+str(args.npoints)+'.npy'),
                               batch_size=args.test_batchsize,
                               shuffle=False,
                               **kwargs)

#.......................Define network........................................
class PairwiseInteractions(nn.Module):
    """ Pairwise interactions block taking speeds into account.
    """
    def __init__(self, d_x, d_v, d_out, N):
        super(PairwiseInteractions, self).__init__()
        self.npoints = args.npoints
        self.d_x = d_x # underlying dimension of input measure
        self.d_v = d_v # for input speeds
        self.d_out = d_out # underlying dimension of output measure
        self.N = N
        self.meas_x = nn.Linear(2*self.d_x+2*self.d_v, self.d_out)

    def forward(self, x0, v0):
        batch_size = x0.size(0)
        # compute pairwise distances for nearest neighbor search.
        distances = torch.sqrt(batch_Lpcost(x0,x0,2,self.d_x))
        # select N nonzero interactions of interest per point.
        val, idx = torch.topk(distances,self.N,2,largest=False,sorted=True)
        distances = None
        val = None
        # tensorized measure of size (batch_size,(N-1)*npoints,2*d) and corresponding speeds
        x0 = batch_index_select_NN(x0.view(batch_size,self.npoints,self.d_x),idx)
        v = batch_index_select_NN(v0.view(batch_size,self.npoints,self.d_v),idx)
        x = torch.cat((x0,v),2)
        v = None
        # batch multiplication with weights to create new measure x_new.
        x = self.meas_x(x)
        x = F.relu(x)
         # sum over neighbors to create new measure of size (batch_size,npoints,d_out)
        x = x.view(batch_size,args.npoints,self.N-1,self.d_out)
        x = torch.sum(x,2).view(batch_size,args.npoints*self.d_out)
        x /= (self.N-1)
        return x, v0


class Net(nn.Module):
    def __init__(self, d, d_in, N):
        super(Net, self).__init__()
        self.d = d
        self.d_in = d_in
        self.npoints = args.npoints
        self.N = N
        self.eb1 = PairwiseInteractions(d_x=2, d_v=2, d_out=30, N=self.N)
        self.eb2 = PairwiseInteractions(d_x=30, d_v=2, d_out=60, N=self.N)
        self.eb3 = PairwiseInteractions(d_x=60, d_v=2, d_out=2, N=self.N)

    def forward(self, x, v):
        x0 = x
        x, v = self.eb1(x,v)
        x, v = self.eb2(x,v)
        x, v = self.eb3(x,v)
        return x0-x

model = Net(d=2, d_in=2, N=args.npoints).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

#.....................Training/test steps.....................................
def train(epoch):
    model.train()
    '''
    # stochastic version
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data[:,:args.d*args.npoints], data[:,args.d*args.npoints:])
    mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((data.size(0),args.npoints))
    loss = 2*log_sinkhorn_batch(target,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(target,target,args.d,mes,mes,args.eps,args.maxiter,2)
    loss /= data.size(0)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
    '''
    # training on whole dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data[:,:args.d*args.npoints], data[:,args.d*args.npoints:])
        mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((data.size(0),args.npoints))
        loss = 2*log_sinkhorn_batch(target,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(target,target,args.d,mes,mes,args.eps,args.maxiter,2)
        loss /= data.size(0)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
    
    
def test(epoch):
    model.eval()
    test_loss = 0
    test_loss_previous = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data[:,:args.d*args.npoints], data[:,args.d*args.npoints:])
            #if epoch % args.log_interval == 0:
            #    np.save('original-'+str(epoch), target.data.cpu().numpy())
            #    np.save('reconstruction-'+str(epoch), output.data.cpu().numpy())
            mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((output.size(0),args.npoints))
            test_loss += 2*log_sinkhorn_batch(target,output,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(target,target,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2).item()
            output = data[:,:args.d*args.npoints]
            test_loss_previous += 2*log_sinkhorn_batch(target,output,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(target,target,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2).item()
            
    test_loss /= len(test_loader.dataset)
    test_loss_previous /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.7f}'.format(test_loss))
    print('Average loss if predicted mu2 is mu1: {:.7f}'.format(test_loss_previous))
    return test_loss

#........................Actually compute training............................
test_loss = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    loss = test(epoch)
    if epoch % args.log_interval == 0:
        loss = test(epoch)
        test_loss.append( loss )
#np.save('test-loss',np.array(test_loss))