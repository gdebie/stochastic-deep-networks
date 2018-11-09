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

#............................Settings.........................................
parser = argparse.ArgumentParser(description='PyTorch classifier')
parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='BStest',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1001, metavar='NEP',
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--npoints', type=int, default=256, metavar='NPTS',
                    help='number of input points per measure')
parser.add_argument('--d', type=int, default=2, metavar='D',
                    help='dimension of input measure')

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
        return normalize_cov(self.x_data[index].view(self.npoints,self.d).float()), self.y_data[index]
    def __len__(self):
        return self.len

# Compute class weights for weighted loss function computation
labels_train = torch.FloatTensor(np.load(path+'datasets/mnist_classif_train.npy'))[:,args.npoints*args.d]
class_sample_count = [ (labels_train == i).sum().item() for i in range(0,10) ]
weights = (1/torch.Tensor(class_sample_count)).type(dtype)

train_loader = data.DataLoader(dataset=MnistCloudDataset(npoints=args.npoints,d=args.d,root='datasets/mnist_classif_train.npy'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               **kwargs)

test_loader = data.DataLoader(dataset=MnistCloudDataset(npoints=args.npoints,d=args.d,root='datasets/mnist_classif_test.npy'),
                               batch_size=args.batch_size,
                               shuffle=True,
                               **kwargs)

#.......................Define network........................................
class ElementaryBlock(nn.Module):
    """ Pairwise interactions block.
    """
    def __init__(self, d, d_out, N, nmoments, first):
        super(ElementaryBlock, self).__init__()
        self.d = d
        self.d_out = d_out
        self.N = N
        self.nmoments = nmoments
        self.meas_x = nn.Linear(2*self.d, self.d_out)
        self.vect_x = nn.Linear(2*self.d, self.nmoments)
        if first:
            self.meas_z = nn.Linear(1, self.d_out)
            self.vect_z = nn.Linear(1, self.nmoments)
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

class Net(nn.Module):
    def __init__(self, N, nmoments, hidden_dim, hidden_dim2):
        super(Net, self).__init__()
        self.N = N
        self.nmoments = nmoments
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.eb1 = ElementaryBlock(d=2, d_out=50, N=self.N, nmoments=self.nmoments, first=True)
        self.eb2 = ElementaryBlock(d=50, d_out=1, N=self.N, nmoments=self.nmoments, first=False)
        self.fc1 = nn.Linear(self.nmoments, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim2)
        self.fc3 = nn.Linear(self.hidden_dim2, 10)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim2, momentum=0.1)

    def forward(self, x):
        z = torch.zeros(x.size(0),1).type(dtype)
        x, z = self.eb1(x,z)
        x, z = self.eb2(x,z)
        x = None
        z = F.relu(self.bn1(self.fc1(z)))
        z = F.dropout(z, p=0.7, training=self.training)
        z = F.relu(self.bn2(self.fc2(z)))
        z = F.dropout(z, p=0.7, training=self.training)
        z = self.fc3(z)
        return F.log_softmax(z, dim=1)

model = Net(N=30, nmoments=1000, hidden_dim=800, hidden_dim2=400).to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

#.....................Training/test steps.....................................
def train(epoch):
    model.train()
    '''
    # stochastic version
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target.long(), weight=weights)
    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(
        epoch, loss.item()))
    '''
    # training on whole dataset
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target.long(), weight=weights)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target.long(), size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * accuracy))
    return test_loss, accuracy

#........................Actually compute training............................
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 45, gamma=0.75)
test_loss = []
test_accuracy = []
for epoch in range(1, args.epochs + 1):
    scheduler.step()
    train(epoch)
    if epoch % args.log_interval == 0:
        loss, accuracy = test()
        test_loss.append( loss )
        test_accuracy.append( accuracy )
np.save('test-loss',np.array(test_loss))
np.save('test-accuracy',np.array(test_accuracy))