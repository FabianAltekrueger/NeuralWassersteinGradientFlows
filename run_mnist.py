# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#2301.11624
#
# Please cite the paper, if you use the code.
# 
# The script provides the example for the discrepancy with MNIST as target.

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
import torchvision.datasets as td
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import utils
import schemes.NeuralBackwardScheme as backward
import schemes.NeuralForwardScheme as forward
import schemes.ParticleFlow as particle_flow


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--flow', default = 'backward',
                    help='Choose between neural backward scheme (backward), neural forward scheme (forward) and particle flow (particle) (default: neural backward scheme)')
args = parser.parse_args()

def save_image(trajectory,name):
    grid = make_grid(trajectory,nrow=m,padding=1,pad_value=.5)
    plt.imsave(f'discrepancy/MNIST/{name}.png',torch.clip(grid.permute(1,2,0),0,1).cpu().numpy())
    return

if __name__ == '__main__':
    taus = [0.5,1.,2.,3.,6.]
    
    #experiment configurations
    r=1
    d=28**2
    m = 25 #number samples
    final_time = 48000
    time = 0
    
    
    mnist = td.MNIST('mnist',transform=transforms.ToTensor(),download=True)
    number_s = 100
    data = DataLoader(dataset=mnist,batch_size=number_s)
    target = next(iter(data))[0].view(number_s,28**2).to(device)
    
    #network configuration
    hidden_layers = 2
    nodes = 2048
    learning_rate = 1e-3
    iterations = {'it_start':5000, 'it':2000, 'start':1}
    batch = m
    latent_dim = d
    usekeops = False
    
    if args.flow in ['backward', 'neural backward scheme', 'backward scheme']:
        name = 'backward_scheme'
        particles = 0.5 * torch.ones((m,d),dtype=dtype)
        func = lambda p1,p2,t,r,usekeops: backward.interaction_energy_term(p1,p2,t,r,usekeops) + backward.potential_energy_term(p1,p2,t,r,usekeops)
        flow = backward.NeuralBackwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers, usekeops = usekeops,
                        target = target)
        
    elif args.flow in ['forward', 'neural forward scheme', 'forward scheme']:
        batch1 = 2000
        name = 'forward_scheme'
        particles = 0.5 * torch.ones((m,d),dtype=dtype)
        func = lambda p1,p2,t1,t2,target,r: forward.interaction_energy_term(p1,p2,t1,t2,target,r)[1] + forward.potential_energy_term(p1,p2,t1,t2,target,r)[1]
        flow = forward.NeuralForwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, batch1 = batch1, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers, target = target)
  
    elif args.flow in ['particle flow', 'particle', 'particle gradient flow']:
        name = 'particle_flow'
        particles = 10**(-9) * (torch.rand((m,d),dtype=torch.double) - 0.5) + torch.tensor(0.5)
        func = lambda p,t,r,usekeops: particle_flow.interaction_energy_term(p,t,r,usekeops) + particle_flow.potential_energy_term(p,t,r,usekeops)
        flow = particle_flow.ParticleFlow(r = r, functional = func, usekeops = usekeops, target = target)
        
    else:
        print('The scheme is not known.')
        exit()
    
    print(f'Compute {name} for dicrepancy with MNIST as target for r={r}, d={d}')
    
    trajectory = torch.empty(0,device=device)
    for i in tqdm(range(int(5/taus[0] + 25/taus[1] + 50/taus[2] + 6000/taus[3] + (final_time-6000)/taus[-1]))):
        tau = taus[0]
        if time >=5:
            tau = taus[1]
        if time >=25:
            tau = taus[2]
        if time >50:
            tau = taus[3]
        if time >=6000:
            tau = taus[4]  
        #save scattered particles as image
        if round(time,2) in [0,1,2,20,150] or round(time,1)%600 == 0:
            trajectory = torch.cat([trajectory,particles.reshape(m,1,28,28).to(device).to(dtype)],dim=0)
            save_image(trajectory,name)
        
        #apply step
        particles, time = flow.apply_step(particles, tau, time)
    
    save_image(particles,time,name,d,r)
