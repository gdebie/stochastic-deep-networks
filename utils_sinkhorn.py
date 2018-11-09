from __future__ import division, print_function, absolute_import
import torch
import torch.nn.functional as nn
import torch.autograd as autograd
import torch.optim as optim
import argparse
import numpy as np
from math import pi
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch classif, VAE and flocking utils')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.cuda:
    dtype = torch.cuda.FloatTensor
    itype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    itype = torch.LongTensor


#..................Weight and biases initialisation...........................
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.randn(*size) * xavier_stddev


#........................normalization steps..................................
def normalize_block(x,d):
    """ Normalizes batch of tensors to fit in unit box.
    """
    batch_size,n_input = x.size()
    n = int(n_input/d)
    x = x.view(batch_size,n,d)
    v_c = torch.sum(x,1)/n
    x = x - v_c.unsqueeze(1).repeat(1,n,1) + 1/2*torch.ones((batch_size,n,d))
    lambd = torch.min(0.5/(torch.max(torch.max(x,1)[0],1)[0]-0.5),-0.5/(torch.min(torch.min(x,1)[0],1)[0]-0.5)).unsqueeze(1).unsqueeze(1).repeat(1,n,d)
    return (torch.mul(lambd,x) + (1-lambd)/2).view(batch_size,n_input)

def normalize_cov(x):
    ''' Computes covariance normalization.
    '''
    n, d = x.size()
    x = x.t()
    x = x - torch.mean(x, 1).unsqueeze(1).repeat(1,x.size(1))
    cov = torch.matmul(x,x.t())/(n-1)
    whiten = torch.diag(torch.diag(cov)**(-0.5))
    return torch.matmul(whiten,x).t().contiguous().view(n*d)    
    

#...............standard sinkhorn stuff in batch form..........................
def batch_Lpcost(t1,t2,p,d):
    """ Yields pairwise cost matrix d(t1_i,t2_j)**p for each element of batch.
    """
    batch_size, n_input = t1.size()
    batch_size, m_input = t2.size()
    n,m = int(n_input/d), int(m_input/d)
    t1 = t1.contiguous().view(batch_size,n,d)
    t2 = t2.contiguous().view(batch_size,m,d)
    return torch.sum((torch.abs(t1.unsqueeze(3)-t2.transpose(1,2).unsqueeze(1)))**p,2)

def diag(A):
    """ Taking torch.diag(vec) along dimension 1, adding a dimension.
    """
    batch_size, n = A.size()
    return torch.mul(A.unsqueeze(2).repeat(1,1,n), Variable(torch.eye(n).repeat(batch_size,1,1).type(dtype)))

def M(a,b,C,eps):
    batch_size,n = a.size()
    batch_size,m = b.size()
    return (-C + a.unsqueeze(2).repeat(1,1,m) + b.unsqueeze(1).repeat(1,n,1)) / eps

def lse(M):
    return torch.log(torch.exp(M).sum(2, keepdim = True)+1e-6)

def log_sinkhorn_batch(t1,t2,d,mu,nu,eps,maxiter,p):
    """ Log-sinkhorn algorithm in batch form, computes batch_size 
        Sinkhorns in parallel.
    """
    batch_size, n_input = t1.size()
    batch_size, m_input = t2.size()
    n,m = int(n_input/d), int(m_input/d)
    C = batch_Lpcost(t1,t2,p,d) 
    a = Variable( torch.zeros(batch_size,n).type(dtype) )
    b = Variable( torch.zeros(batch_size,m).type(dtype) )
    t1 = t1.view(batch_size,n,d)
    t2 = t2.view(batch_size,m,d)
    for i in range(maxiter):
        a = a + eps * (torch.log(mu) - lse(M(a,b,C,eps)).squeeze())
        b = b + eps * (torch.log(nu) - lse(M(a,b,C,eps).transpose(1,2)).squeeze())
    return torch.sum(torch.mul(torch.exp(M(a,b,C,eps)),C))


#...................Elementary block related functions.........................
def batch_index_select_1percloud(x,idx):
    """ x has shape (batch_size,n,d) and idx size (batch_size).
        Selects idx[i]-th point from x[i]-th cloud.
        Ending up with 1 point per cloud, i.e. size (batch_size,1,d).
    """
    batch_size, n, d = x.size()
    return torch.gather(x,1,idx.unsqueeze(1).unsqueeze(1).repeat(1,n,d))[:,0,:].unsqueeze(1)

def batch_index_select_Npercloud(x,idx):
    """ x has shape (batch_size,n,d) and idx size (batch_size,N).
        Selects N points at indexes idx[i] from x[i]-th cloud.
        Ending up with N points per cloud, i.e. size (batch_size,N,d).
    """
    batch_size, n, d = x.size()
    N = idx.size(1)
    return torch.gather(x.unsqueeze(2).repeat(1,1,N,1),1,idx.unsqueeze(1).unsqueeze(3).repeat(1,n,1,d))[:,0,:,:]

def batch_index_select_nNpercloud(x,idx):
    """ x has shape (batch_size,n,d) and idx size (batch_size,n,N).
        Selects N points at indexes idx[i,j] from point j of x[i]-th cloud.
        Ending up with n*N points per cloud, i.e. size (batch_size,N*n,d).
    """
    batch_size, n, d = x.size()
    N = idx.size(2)
    return torch.gather(x.unsqueeze(2).repeat(1,1,N,1),1,idx.unsqueeze(3).repeat(1,1,1,d)).view(batch_size,n*N,d)

def batch_index_select_NN(x,idx):
    """ Agglomerates initial points and their selected nearest neighbors
        (ending up in dimension 2*d).
        Inputs:
        x: has shape (batch_size,n,d), initial point cloud;
        idx: has shape(batch_size,n,N), nearest neighbors of each point
        from cloud.
        Outputs: tensorized measure of size (batch_size,(N-1)*n,2*d), which
        represents pairwise interactions between points and each of their
        neighbors.
    """
    batch_size, n, d = x.size()
    N = idx.size(2)
    x_pairs = batch_index_select_nNpercloud(x,idx[:,:,1:])
    return torch.cat([x.repeat(1,1,N-1).view(batch_size,(N-1)*n,d), x_pairs], 2).view(batch_size,(N-1)*n,2*d)


#..................farthest point sampling main function......................
def batch_farthest_sampling_for(x,n_sub,d):
    """ Subsamples x to end up with n_sub points per cloud through
        farthest point sampling.
        Inputs: cloud x of size (batch_size,n*d) and subsampling strength n_sub.
        Outputs: new subsampled batch of size (batch_size,n_sub*d).
    """
    batch_size, n_input = x.size()
    n = int(n_input/d)
    if n <= n_sub:
        return x
    else:
        # compute pairwise distances.
        distances = torch.sqrt(batch_Lpcost(x,x,2,d))
        x = x.view(batch_size,n,d)
        result = torch.zeros(batch_size,n_sub,d).type(dtype)
        for i in range(n_sub):
            # Determine maximum of minimal nonzero distances
            val, idx = torch.topk(distances,i+2,dim=1,largest=False)
            val, idx = torch.topk(val[:,-1,:],1,dim=1,largest=True)
            idx = idx.squeeze(1)
            # Delete corresponding row and column in distances matrix,
            # for that point not to be chosen twice.
            delete_col = batch_index_select_1percloud(distances,idx) # size (batch_size,1,n)
            distances = distances - (distances == delete_col).float()*delete_col
            delete_row = batch_index_select_1percloud(distances.transpose(1,2),idx)
            distances = distances.transpose(1,2) - (distances.transpose(1,2) == delete_row).float()*delete_row
            distances = distances.transpose(1,2)
            # Store result
            result[:,i,:] = batch_index_select_1percloud(x,idx).squeeze(1)
        return result.view(batch_size,n_sub*d)