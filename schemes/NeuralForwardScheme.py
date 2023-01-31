# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#23xx.xxxxx
#
# Please cite the paper, if you use the code.
# 
# The script provides the neural forward scheme.

import torch

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

def differences_forward(p,p_der,q,q_der):
    '''
    Compute the pairwise differences
    '''
    dim = p.shape[1]
    m_p, m_q = p.shape[0], q.shape[0]
    diff = p.reshape(m_p,1,dim) - q.reshape(1,m_q,dim)
    diff_der = p_der.reshape(m_p,1,dim) - q_der.reshape(1,m_q,dim)
    return diff, diff_der

def distance_forward(p,p_der,q,q_der):
    '''
    Compute the norms of the pairwise differences of the particles and
    the derivate
    '''    
    dim = p.shape[1]
    diff, diff_der = differences_forward(p,p_der, q,q_der)
    diff = torch.reshape(diff,[-1,dim])
    diff_der = torch.reshape(diff_der,[-1,dim])
    out = torch.linalg.vector_norm(diff,ord=2,dim=-1)
    out_der = torch.zeros_like(out)
    is_zero = out<1e-10
    not_zero = torch.logical_not(is_zero)
    out_der[is_zero] = torch.linalg.vector_norm(diff_der[is_zero],ord=2,dim=-1)
    out_der[not_zero] = (torch.sum(diff[not_zero]*diff_der[not_zero],-1)/out[not_zero])
    return out,out_der

def energy_forward(p,p_der,q,q_der,r=1.):
    '''
    Sum up over all computed distances and compute the derivate
    '''
    dist, dist_der = distance_forward(p,p_der,q,q_der)
    out = 0.5*torch.sum(dist**r)/(p.shape[0]*q.shape[0])
    dist_power_r_der = r*dist_der*dist**(r-1)
    out_der = 0.5*torch.sum(dist_power_r_der)/(p.shape[0]*q.shape[0])
    return out, out_der

def other_energy_term(p,p_der,q,q_der,dim,r=1.,point=0):
    '''
    Compute the energy functional and its derivate introduced in Appendix F,Example 3, given by
    F(\mu) = \int_{\R^2} 1_{(-\infty,0)}(x) |y| - x  \dx \mu(x,y) 
            - 1/2 \int_{\R^2} \int_{\R^2} 1_{[0,\infty)^2}(x_1,x_2) |y_1 - y_2| \dx \mu(x_1,y_2) \dx \mu(x_2,y_2).
    '''
    ps = torch.cat([p,q],dim=0)
    qs = torch.cat([p_der,q_der],dim=0)
    
    implosion_term = (1/ps.shape[0]) * torch.sum(torch.abs(ps[ps[:,0]<point][:,1]))
    implosion_term += (1/ps.shape[0]) * torch.sum(-ps[:,0])
    
    relevant_ps = ps[ps[:,0]<point][:,1]
    implosion_der = (1/ps.shape[0]) * torch.sum(torch.sign(relevant_ps)*qs[ps[:,0]<point][:,1])
    implosion_der += (1/ps.shape[0]) * torch.sum(-qs[:,0])
    
    loss = implosion_term
    loss_der = implosion_der
    relevant_p = p[p[:,0]>=0][:,1].unsqueeze(1)
    relevant_q = q[q[:,0]>=0][:,1].unsqueeze(1)
    relevant_p_der = p_der[p1[:,0]>=0][:,1].unsqueeze(1)
    relevant_q_der = q_der[q[:,0]>=0][:,1].unsqueeze(1)

    if relevant_p.shape[0]>0 and relevant_q.shape[0]>0:
        explosion_term, explosion_der = energy_forward(relevant_p,relevant_p_der,relevant_q,relevant_q_der,r=r)
        loss += -explosion_term  * (relevant_p.shape[0] * relevant_q.shape[0])/(p.shape[0]*q.shape[0])
        loss_der += -explosion_der  * (relevant_p.shape[0] * relevant_q.shape[0])/(p.shape[0]*q.shape[0])
    return loss, loss_der

def interaction_energy_term(particles1,particles2,tangent1,
                                    tangent2,target_particles,r=1.):
    '''
    Compute the interaction energy and its derivative
    '''                                    
    out_value, out_derivative = energy_forward(particles1,tangent1,particles2,tangent2,r=r)
    return -out_value, -out_derivative

def potential_energy_term(particles1,particles2,tangent1,
                                    tangent2,target_particles,r=1.):
    '''
    Compute the potential energy and its derivative
    '''                                    
    out_value, out_derivative = energy_forward(torch.cat([particles1,particles2],dim=0),torch.cat([tangent1,tangent2],dim=0),
                            target_particles,torch.zeros_like(target_particles),r=r)
    return 2*out_value, 2*out_derivative
    

def velocity_plan_norm(particles,tangent,squared=False):
    '''
    Compute the norm of the velocity plan
    '''
    if squared:
        out=torch.sum(tangent**2)/tangent.shape[0]
    else:
        out=torch.linalg.vector_norm(tangent)/(tangent.shape[0]**0.5)
    return out


class NeuralForwardScheme:
    '''
    Defines the class for computing the gradient flow with respect to
    the given functional via the forward scheme
    '''
    def __init__(self, r, functional, learning_rate, it, batch, batch1, dim, 
                latent_dim, sub_net_size, hidden_layers, target = None):
        self.r = r
        self.network = get_network(dim = dim, latent_dim = latent_dim,
                    sub_net_size = sub_net_size, hidden_layers = hidden_layers)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = learning_rate)
        self.functional = functional
        self.iter = it['it']
        self.iter_start = it['it_start']
        self.start = it['start']
        self.batch = batch #batch size for training
        self.batch1 = batch1 #batch size for forward step
        self.latent_dim = latent_dim
        self.target = target
        
    def apply_step(self, particles, dt, time):
        '''
        first train the Markov kernel, then compute forward step
        '''
        if not torch.is_tensor(particles):
            particles_ten = torch.tensor(particles,dtype=torch.float,device=device)
        else:
            particles_ten = particles.to(device)
        steps = self.iter
        if time<self.start:
            steps = self.iter_start
        from tqdm import tqdm
        for i in tqdm(range(steps)):
            self.optimizer.zero_grad()
            perm1 = torch.randperm(particles_ten.shape[0])[:self.batch]
            perm2 = torch.randperm(particles_ten.shape[0])[:self.batch]
            particles1 = particles_ten[perm1]
            particles2 = particles_ten[perm2]
            tangent1 = self.network(torch.cat((particles1,torch.randn((particles1.shape[0],self.latent_dim),device=device)),1))
            tangent2 = self.network(torch.cat((particles2,torch.randn((particles2.shape[0],self.latent_dim),device=device)),1))

            numerator = self.functional(particles1,particles2,tangent1,tangent2,self.target,r=self.r)
            denominator = velocity_plan_norm(torch.cat([particles1,particles2],dim=0),torch.cat([tangent1,tangent2],dim=0))

            loss = numerator/(denominator + 1e-2)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            perm1 = torch.randperm(particles_ten.shape[0])[:self.batch1]
            perm2 = torch.randperm(particles_ten.shape[0])[:self.batch1]
            particles1 = particles_ten[perm1]
            particles2 = particles_ten[perm2]
            tangent1 = self.network(torch.cat((particles1,torch.randn((particles1.shape[0],self.latent_dim),device=device)),1))
            tangent2 = self.network(torch.cat((particles2,torch.randn((particles2.shape[0],self.latent_dim),device=device)),1))
            Dv_hat = self.functional(particles1,particles2,tangent1,tangent2,
                                        self.target, r=self.r)
                                        
            Dv_hat = torch.max(-Dv_hat,torch.tensor(0.,dtype=torch.float,device=device))
            tangent = self.network(torch.cat((particles_ten,torch.randn((particles_ten.shape[0],self.latent_dim),device=device)),1))
            v_hat_norm_squared = velocity_plan_norm(torch.cat([particles1,particles2],dim=0),torch.cat([tangent1,tangent2],dim=0),squared=True)
            
            particles_out = particles_ten + dt*(Dv_hat/v_hat_norm_squared)*tangent
            particles_out = particles_out.detach()
        return particles_out, time+dt
