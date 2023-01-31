# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#23xx.xxxxx
#
# Please cite the paper, if you use the code.
# 
# The script provides the particle flow.

import torch
from pykeops.torch import LazyTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.double

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
        out=torch.sqrt(torch.sum(diff*diff, dim=2))
    return out
    
def energy_grad(p,q,r=1.,usekeops = False):
    '''
    Compute the gradient of the energy term
    '''
    mp, mq = p.shape[0], q.shape[0]
    if usekeops:
        diff = differences_keops(p,q)
        dist = distance(p,q,diff=diff,usekeops=usekeops) + 1e-13
        grad = r * diff / dist**(2-r)
        out=grad.sum(1)
    else:
        diff = differences(p,q)
        dist = distance(p,q).reshape(mp, mq, 1)
        dist[dist==0] = 1
        grad = r * diff / dist**(2-r)
        out = torch.sum(grad, dim=1)
    return out

def interaction_energy_term(particles,target_particles,r,usekeops=False):
    '''
    Compute the gradient of the interaction energy term
    '''
    m = particles.shape[0]
    return -energy_grad(particles,particles,r,usekeops=usekeops) / m**2
    
def potential_energy_term(particles,target_particles,r,usekeops=False):
    '''
    Compute the gradient of the interaction energy term
    '''
    m = particles.shape[0]
    return energy_grad(particles,target_particles,r,usekeops=usekeops) / (m * target_particles.shape[0])

class ParticleFlow:
    '''
    Defines the class for computing the gradient flow with respect to
    the given functional via the particle flow
    '''    
    def __init__(self, r, functional, target = None, usekeops = False):
        self.r = r
        self.functional = functional
        self.target = target.double() if target is not None else target
        self.usekeops = usekeops
        
    def apply_step(self, particles, dt, time):
        if not torch.is_tensor(particles):
            particles_ten = torch.tensor(particles,dtype=dtype,device=device)
        else:
            particles_ten = particles.to(device)
        
        m = particles.shape[0]
        grad = self.functional(particles_ten,self.target,self.r,usekeops=self.usekeops)
        return particles_ten - dt * m * grad, time + dt
