# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#2301.11624
#
# Please cite the paper, if you use the code.
# 
# The script provides the example for the discrepancy with Dirac measures as target.

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse


import schemes.NeuralBackwardScheme as backward
import schemes.NeuralForwardScheme as forward
import schemes.ParticleFlow as particle_flow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--flow', default = 'backward',
                    help='Choose between neural backward scheme (backward), neural forward scheme (forward) and particle flow (particle) (default: neural backward scheme)')
parser.add_argument('--tau', type = float, default = 0.1,
                    help='time step size tau (default: tau=0.1)')
args = parser.parse_args()

def save_scatter(particles,time,name,d,r):
    matplotlib.rcParams.update({'font.size': 25})
    particles = particles.cpu()
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = plt.axes(xlim=(-2.3,2.3),ylim=(-2.3,2.3))
    ax1.set_aspect('equal', adjustable='box')
    size1 = 100/(m**0.5)*torch.ones(m,dtype=float)
    scatter1=ax1.scatter(particles[:,0], particles[:,1])
    scatter1.set_sizes(size1)
    time_text = ax1.text(0.05, .95, f'time = {time:.1f}', horizontalalignment='left', 
                            verticalalignment='top', transform=ax1.transAxes)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.savefig(f'discrepancy/dirac/{name}_{time:.1f}.png',dpi=100,bbox_inches='tight')
    plt.close()
    return

if __name__ == '__main__':
    tau = args.tau
    
    #experiment configurations
    r=1
    d=2
    m = 2000 #number samples
    final_time = 40
    time = 0
    target = torch.tensor([[1,1],[-1,-1]],dtype=dtype,device=device)
        
    #network configuration
    hidden_layers = 4
    nodes = 128
    learning_rate = 1e-3
    iterations = {'it_start':4000, 'it':2000, 'start':1}
    batch = m
    latent_dim = d
    usekeops = False
    
    if args.flow in ['backward', 'neural backward scheme', 'backward scheme']:
        name = 'backward_scheme'
        particles = torch.zeros((int(m/2),d),dtype=dtype) - torch.tensor([1,-1])
        particles = torch.cat([particles,torch.zeros((int(m/2),d),dtype=dtype) - torch.tensor([-1,1])],dim=0)
        func = lambda p1,p2,t,r,usekeops: backward.interaction_energy_term(p1,p2,t,r,usekeops) + backward.potential_energy_term(p1,p2,t,r,usekeops)
        flow = backward.NeuralBackwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers, usekeops = usekeops,
                        target = target)
        
    elif args.flow in ['forward', 'neural forward scheme', 'forward scheme']:
        batch1 = 2000
        name = 'forward_scheme'
        particles = torch.zeros((int(m/2),d),dtype=dtype) - torch.tensor([1,-1])
        particles = torch.cat([particles,torch.zeros((int(m/2),d),dtype=dtype) - torch.tensor([-1,1])],dim=0)
        func = lambda p1,p2,t1,t2,target,r: forward.interaction_energy_term(p1,p2,t1,t2,target,r)[1] + forward.potential_energy_term(p1,p2,t1,t2,target,r)[1]
        flow = forward.NeuralForwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, batch1 = batch1, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers, target = target)
  
    elif args.flow in ['particle flow', 'particle', 'particle gradient flow']:
        name = 'particle_flow'
        particles = 10**(-9) * (torch.rand((int(m/2),d),dtype=torch.double) - 0.5) - torch.tensor([1,-1])
        particles = torch.cat([particles,10**(-9) * (torch.rand((int(m/2),d),dtype=torch.double) - 0.5) - torch.tensor([-1,1])],dim=0)
        func = lambda p,t,r,usekeops: particle_flow.interaction_energy_term(p,t,r,usekeops) + particle_flow.potential_energy_term(p,t,r,usekeops)
        flow = particle_flow.ParticleFlow(r = r, functional = func, usekeops = False, target = target)
        
    else:
        print('The scheme is not known.')
        exit()
    
    print(f'Compute {name} for dicrepancy with two Dirac measures as target for r={r}, d={d} and tau={tau}')
    
    for i in tqdm(range(int(np.ceil(final_time/tau)))):
        #save scattered particles as image
        if i%10 == 0:
            save_scatter(particles,time,name,d,r)
        
        #apply step
        particles, time = flow.apply_step(particles, tau, time)
    
    save_scatter(particles,time,name,d,r)
        
    
