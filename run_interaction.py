# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#2301.11624
#
# Please cite the paper, if you use the code.
# 
# The script provides the example for the interaction energy.

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy.special import gamma

import schemes.NeuralBackwardScheme as backward
import schemes.NeuralForwardScheme as forward
import schemes.ParticleFlow as particle_flow

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float

parser = argparse.ArgumentParser()
parser.add_argument('--flow', default = 'backward',
                    help='Choose between neural backward scheme (backward), neural forward scheme (forward) and particle flow (particle) (default: neural backward scheme)')
parser.add_argument('--r', type = float, default = 1,
                    help='exponent of the Riesz kernel (default: r=1)')
parser.add_argument('--d', type = int, default = 2,
                    help='dimension of the image space (default: d=2)')
parser.add_argument('--tau', type = float, default = 0.05,
                    help='time step size tau (default: tau=0.05)')
args = parser.parse_args()

def save_scatter(particles,time,name,d,r):
    matplotlib.rcParams.update({'font.size': 25})
    particles = particles.cpu()
    if d == 2:
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = plt.axes(xlim=(-1.,1.),ylim=(-1.,1.))
        size1 = 100/(m**0.5)*torch.ones(m,dtype=float)
        scatter1=ax1.scatter(particles[:,0], particles[:,1])
        scatter1.set_sizes(size1)
        time_text = ax1.text(0.05, .95, f'time = {time:.1f}', horizontalalignment='left', 
                                verticalalignment='top', transform=ax1.transAxes)
                        
        circle1 = plt.Circle((0, 0), 0, fill=False)
        ax1.add_patch(circle1)
        s = ( (gamma(2-(r/2)) * gamma((d+r)/2) * r) / ((d/2) * gamma(d/2)) )**(1/(2-r))
        r_t = time**(1/(2-r)) * s * (2-r)**(1/(2-r))
        circle1.set_center((0, 0))
        circle1.set_radius(r_t)
        plt.axis('off')
        plt.savefig(f'interaction/{name}_{r}r{d}d_{time:.1f}.png',dpi=100,bbox_inches='tight')
        plt.close()
    return

if __name__ == '__main__':
    r = args.r
    d = args.d
    tau = args.tau
    
    #experiment configurations
    m = 2000 #number samples
    final_time = 1.2
    time = 0
    
    #network configuration
    hidden_layers = 1
    nodes = 64
    learning_rate = 1e-3
    iterations = {'it_start':5000, 'it':5000, 'start':1}
    batch = 100
    latent_dim = d
    usekeops = False
    
    if args.flow in ['backward', 'neural backward scheme', 'backward scheme']:
        name = 'backward_scheme'
        particles = torch.zeros(m,d)
        func = backward.interaction_energy_term
        flow = backward.NeuralBackwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers, usekeops = usekeops)
        
    elif args.flow in ['forward', 'neural forward scheme', 'forward scheme']:
        batch1 = 2000
        name = 'forward_scheme'
        particles = torch.zeros(m,d)
        func = lambda p1,p2,t1,t2,target,r: forward.interaction_energy_term(p1,p2,t1,t2,target,r)[1]
        flow = forward.NeuralForwardScheme(r=r,functional = func, learning_rate = learning_rate,
                        it = iterations, batch = batch, batch1 = batch1, dim = d, latent_dim = latent_dim,
                        sub_net_size = nodes, hidden_layers = hidden_layers)
  
    elif args.flow in ['particle flow', 'particle', 'particle gradient flow']:
        name = 'particle_flow'
        particles = 10**(-9) * (torch.rand((m,d),dtype = torch.double) - 0.5)
        func = particle_flow.interaction_energy_term
        flow = particle_flow.ParticleFlow(r=r,functional=func,usekeops=usekeops)
        
    else:
        print('The scheme is not known.')
        exit()
    
    print(f'Compute {name} for interaction energy for r={r}, d={d} and tau={tau}')
    
    for i in tqdm(range(int(np.ceil(final_time/tau)))):
        #save scattered particles as image
        if i%4 == 0:
            save_scatter(particles,time,name,d,r)
        
        dt = tau
        #smaller step size for exponent r<1
        if r < 1:
            slow_start = 8
            dt = min(10**(-slow_start+i),tau)
            if i == slow_start - 1:
                dt = 0.05 - time
        #apply step
        particles, time = flow.apply_step(particles, dt, time)
    
    save_scatter(particles,time,name,d,r)    

