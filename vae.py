from __future__ import print_function
import argparse
from utils_sinkhorn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

#........................Input path...........................................
OnSever = False
if OnSever:
    path = ''
else:
    path = ''

#...................Training settings.........................................
parser = argparse.ArgumentParser(description='PyTorch MNIST VAE')
parser.add_argument('--batch-size', type=int, default=124, metavar='BS',
                    help='input batch size for training (default: 32)')
parser.add_argument('--epochs', type=int, default=1001, metavar='NEP',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--npoints', type=int, default=100, metavar='NPTS',
                    help='number of input points per measure')
parser.add_argument('--d', type=int, default=2, metavar='D',
                    help='dimension of input measure')
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

#........................Load dataset.........................................
class MnistCloudDataset(data.Dataset):
    """ MNIST Point Cloud Dataset.
    """
    def __init__(self, npoints, d, root):
        self.d = d
        self.npoints = npoints
        self.root = root
        xy = np.load(path+self.root)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:self.d*self.npoints])
        self.y_data = torch.from_numpy(xy[:,self.d*self.npoints])
    def __getitem__(self,index):
        return normalize_block(self.x_data[index].unsqueeze(0).float(),self.d).squeeze(0), self.y_data[index]
    def __len__(self):
        return self.len

train_loader = data.DataLoader(dataset=MnistCloudDataset(npoints=args.npoints,d=args.d,root='datasets/mnist_vae_train.npy'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               **kwargs)

test_loader = data.DataLoader(dataset=MnistCloudDataset(npoints=args.npoints,d=args.d,root='datasets/mnist_vae_test.npy'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               **kwargs)


#.......................Define network........................................
class ElementaryBlock(nn.Module):
    """ Pairwise interactions block.
    """
    def __init__(self, d, d_out, N, nmoments, latent_dim, first, encoder):
        super(ElementaryBlock, self).__init__()
        self.d = d
        self.d_out = d_out
        self.N = N
        self.nmoments = nmoments
        self.latent_dim = latent_dim
        self.meas_x = nn.Linear(2*self.d, self.d_out)
        self.vect_x = nn.Linear(2*self.d, self.nmoments)
        if first and encoder:
            self.meas_z = nn.Linear(1, self.d_out)
            self.vect_z = nn.Linear(1, self.nmoments)
        elif first and not encoder:
            self.meas_z = nn.Linear(self.latent_dim, self.d_out)
            self.vect_z = nn.Linear(self.latent_dim, self.nmoments)
        else:
            self.meas_z = nn.Linear(self.nmoments, self.d_out)
            self.vect_z = nn.Linear(self.nmoments, self.nmoments)

    def forward(self, x, z):
        batch_size = x.size(0)
        # compute pairwise distances for nearest neighbor search.
        distances = torch.sqrt(batch_Lpcost(x,x,2,self.d))
        # select N nonzero interactions of interest per point.
        val, idx = torch.topk(distances,self.N,2,largest=False,sorted=True)
        distances = None
        val = None
        # tensorized measure of size (batch_size,(N-1)*npoints,2*d)
        x = batch_index_select_NN(x.view(batch_size,args.npoints,self.d),idx)
        idx = None
        # batch multiplication with weights to create new measure x_new.
        x_new = self.meas_x(x)
        x_new += self.meas_z(z).unsqueeze(1)
        x_new = F.relu(x_new)
        # sum over neighbors to create new measure of size (batch_size,npoints,d_out)
        x_new = torch.sum(x_new.view(batch_size,args.npoints,self.N-1,self.d_out),2).view(batch_size,args.npoints*self.d_out)
        x_new /= self.N-1
        # batch multiplication with weights to create new vector z_new.
        z_new = self.vect_x(x)
        x = None
        z_new += self.vect_z(z).unsqueeze(1)
        z_new = torch.sum( F.relu( z_new ), 1 )/( (self.N-1)*args.npoints )
        return x_new, z_new


class VAE(nn.Module):
    def __init__(self, N, nmoments, hidden_dim):
        super(VAE, self).__init__()
        self.N = N
        self.nmoments = nmoments
        self.hidden_dim = hidden_dim
        self.latent_dim = 2
        self.n_input = args.npoints*args.d
        ### Encoder...........................................................
        self.eb1 = ElementaryBlock(d=2, d_out=15, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=True, encoder=True)
        self.eb2 = ElementaryBlock(d=15, d_out=30, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=False, encoder=True)
        self.eb3 = ElementaryBlock(d=30, d_out=1, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=False, encoder=True)
        self.fc1 = nn.Linear(self.nmoments, self.hidden_dim)
        self.fc21 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.fc22 = nn.Linear(self.hidden_dim, self.latent_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim, momentum=0.1)
        ### Decoder...........................................................
        self.eb4 = ElementaryBlock(d=2, d_out=15, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=True, encoder=False)
        self.eb5 = ElementaryBlock(d=15, d_out=30, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=False, encoder=False)
        self.eb6 = ElementaryBlock(d=30, d_out=2, N=self.N, nmoments=self.nmoments, latent_dim=self.latent_dim, first=False, encoder=False)
    
    def encode(self, x):
        z = torch.zeros(x.size(0),1).type(dtype)
        x, z = self.eb1(x,z)
        x, z = self.eb2(x,z)
        x, z = self.eb3(x,z)
        x = None
        z = F.relu(self.bn1(self.fc1(z)))
        self.z_mean = self.fc21(z)
        self.z_var = self.fc22(z)
        return self.z_mean, self.z_var

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        x = torch.rand(z.size(0),self.n_input).type(dtype)
        x, z = self.eb4(x,z)
        x, z = self.eb5(x,z)
        x, z = self.eb6(x,z)
        z = None
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE(N=30, nmoments=100, hidden_dim=150).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

#.....................Training/test steps.....................................
def train(epoch):
    model.train()

    '''
    # stochastic version
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output, mu, logvar = model(data)
    mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((data.size(0),args.npoints))
    loss = 2*log_sinkhorn_batch(data,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(data,data,args.d,mes,mes,args.eps,args.maxiter,2)
    loss /= data.size(0)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, loss.item()))
    '''
    # training on whole dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, mu, logvar = model(data)
        mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((data.size(0),args.npoints))
        loss = log_sinkhorn_batch(data,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2) - log_sinkhorn_batch(data,data,args.d,mes,mes,args.eps,args.maxiter,2)
        loss /= data.size(0)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    

def test(epoch):
    model.eval()
    
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, mu, logvar = model(data)
            #np.save('original-'+str(epoch), data.data.cpu().numpy())
            #np.save('reconstruction-'+str(epoch), output.data.cpu().numpy())
            mes = (1./args.npoints*torch.ones(args.npoints).type(dtype)).expand((output.size(0),args.npoints))
            test_loss += 2*log_sinkhorn_batch(data,output,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(data,data,args.d,mes,mes,args.eps,args.maxiter,2).item() - log_sinkhorn_batch(output,output,args.d,mes,mes,args.eps,args.maxiter,2).item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))
    return test_loss

#........................Actually compute training............................
test_loss = []
for epoch in range(1, args.epochs + 1):
    train(epoch)
    if epoch % args.log_interval == 0:
        loss = test(epoch)
        test_loss.append( loss )
    if epoch % 100 == 0:
        # Visualize manifold in 2d from time to time
        a,b = torch.min(model.z_mean).item(), torch.max(model.z_mean).item()
        nx = ny = 15
        x_values = np.linspace(a, b, nx)
        y_values = np.linspace(a, b, ny)

        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                z_mu = torch.FloatTensor( np.array([[xi, yi]]*args.batch_size) ).type(dtype)
                #np.save('epoch-'+str(epoch)+'-i-'+str(i)+'-j-'+str(j), model.decode(z_mu).data.cpu().numpy())
#np.save('test-loss',np.array(test_loss))
