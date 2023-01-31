# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#23xx.xxxxx
#
# Please cite the paper, if you use the code.
# 
# The script provides the neural backward scheme.

import torch
from pykeops.torch import LazyTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float


def get_network(dim,latent_dim,sub_net_size=64,hidden_layers=1):
    '''
    Define the network parametrizing the Markov kernel
    '''
    c_in = dim + latent_dim
    modules = [torch.nn.Linear(c_in, sub_net_size).to(dtype),torch.nn.ReLU()]
    for i in range(hidden_layers):
        modules.append(torch.nn.Linear(sub_net_size, sub_net_size).to(dtype))
        modules.append(torch.nn.ReLU())
    modules.append(torch.nn.Linear(sub_net_size,  dim).to(dtype))
    return torch.nn.Sequential(*modules).to(device)

def W2_term(particles,particles_out):
    '''
    Compute the discretized Wasserstein distance
    '''
    return torch.sum((particles-particles_out)**2)/particles.shape[0]
    
    
def differences_keops(p,q):
    '''
    Compute the pairwise differences using the module pykeops
    '''
    q_k = LazyTensor(q.unsqueeze(1).contiguous())
    p_k = LazyTensor(p.unsqueeze(0).contiguous())
    rmv = q_k-p_k
    return rmv

def differences(p,q):
    '''
    Compute the pairwise differences
    '''
    dim = p.shape[1]
    m_p, m_q = p.shape[0], q.shape[0]
    diff = p.reshape(m_p,1,dim) - q.reshape(1,m_q,dim)
    return diff

def distance(p,q,diff=None,usekeops=False):
    '''
    Compute the norms of the pairwise differences
    '''
    if usekeops:
        if diff is None:
            diff = differences_keops(p,q) + 1e-13
        out = (diff**2).sum(2).sqrt()
    else:
        if diff is None:
            diff = differences(p,q)
        out=torch.linalg.vector_norm(diff,ord=2,dim=2)
    return out

def energy(p,q,r=1.,usekeops=False):
    '''
    Sum up over all computed distances
    '''
    dist = distance(p,q,usekeops=usekeops)
    if usekeops:
        return 0.5*((dist**r).sum(0).sum(0))/(p.shape[0]*q.shape[0])
    else:
        return 0.5*torch.sum(dist**r)/(p.shape[0]*q.shape[0])


def interaction_energy_term(particles_out1,particles_out2,target_particles,r=1.,usekeops=False):
    '''
    Compute the interaction energy
    '''
    return -energy(particles_out1,particles_out2,r=r,usekeops=usekeops)

def potential_energy_term(particles_out1,particles_out2,target_particles,r=1.,usekeops=False):
    '''
    Compute the potential energy
    '''
    return 2*energy(torch.cat([particles_out1,particles_out2],dim=0),target_particles,r=r,usekeops=usekeops)
    
def other_energy_term(particles_out1,particles_out2,target_particles,r=1.,point=0,usekeops=False):
    '''
    Compute the energy functional introduced in Appendix F,Example 3, given by
    F(\mu) = \int_{\R^2} 1_{(-\infty,0)}(x) |y| - x  \dx \mu(x,y) 
            - 1/2 \int_{\R^2} \int_{\R^2} 1_{[0,\infty)^2}(x_1,x_2) |y_1 - y_2| \dx \mu(x_1,y_2) \dx \mu(x_2,y_2).
    '''
    particles = torch.cat([particles_out1,particles_out2],dim=0)
    implosion_term = (1/particles.shape[0]) * torch.sum(torch.abs(particles[particles[:,0]<point][:,1]))
    implosion_term += (1/particles.shape[0]) * torch.sum(-particles[:,0])
    loss = implosion_term
    relevant_y1 = particles_out1[particles_out1[:,0]>=0][:,1].unsqueeze(1)
    relevant_y2 = particles_out2[particles_out2[:,0]>=0][:,1].unsqueeze(1)
    if relevant_y1.shape[0]>0 and relevant_y2.shape[0]>0:
        explosion_term = -energy(relevant_y1,relevant_y2,r=r) * ((relevant_y1.shape[0] * relevant_y2.shape[0])
                    /(particles_out1.shape[0]*particles_out2.shape[0]))
        loss += explosion_term
    return loss
                    
class NeuralBackwardScheme:
    '''
    Defines the class for computing the gradient flow with respect to
    the given functional via the backward scheme
    '''
    def __init__(self, r, functional, learning_rate, it, batch, dim, 
                latent_dim, sub_net_size, hidden_layers, target = None, usekeops = False):
        self.r = r
        self.network = get_network(dim = dim, latent_dim = latent_dim,
                    sub_net_size = sub_net_size, hidden_layers = hidden_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
        self.functional = functional
        self.iter = it['it']
        self.iter_start = it['it_start']
        self.start = it['start']
        self.batch = batch #batch size
        self.latent_dim = latent_dim
        self.target = target
        self.usekeops = usekeops
        
    def apply_step(self, particles, dt, time):
        '''
        first train the Markov kernel, then compute new samples
        '''
        if not torch.is_tensor(particles):
            particles_ten=torch.tensor(particles,dtype=torch.float,device=device)
        else:
            particles_ten = particles.to(device)
        steps = self.iter
        if time<self.start:
            steps = self.iter_start
            
        for i in range(steps):
            self.optimizer.zero_grad()
            perm1 = torch.randperm(particles_ten.shape[0])[:self.batch]
            perm2 = torch.randperm(particles_ten.shape[0])[:self.batch]
            xs1 = particles_ten[perm1]
            xs2 = particles_ten[perm2]
            xs_tmp1 = torch.cat((xs1,torch.randn((xs1.shape[0],self.latent_dim),dtype=dtype,device=device)),1)
            xs_tmp2 = torch.cat((xs2,torch.randn((xs2.shape[0],self.latent_dim),dtype=dtype,device=device)),1)
            particles_out1 = self.network(xs_tmp1)
            particles_out2 = self.network(xs_tmp2)
            
            W2 = W2_term(torch.cat([xs1,xs2],dim=0),torch.cat([particles_out1,particles_out2],dim=0))
            fun_val = self.functional(particles_out1,particles_out2,self.target,r=self.r,usekeops=self.usekeops)
            loss = .5*W2 + dt*fun_val
            loss.backward()
            self.optimizer.step()
            
        particles_ten = torch.cat((particles_ten,torch.randn((particles_ten.shape[0],self.latent_dim),device=device)),1)
        particles_out = self.network(particles_ten)
        particles_out = particles_out.detach()
        return particles_out, time+dt
