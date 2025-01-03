
import numpy as np
import torch
from torch.nn import functional as F


def prox_TV_L1(d, lamb=1.0, maxit=1000,check=100, verbose=0):
       

    # primal and dual step size
    # tau * sigma * L^2 = 1
    L = np.sqrt(12)
    tau = 1/L
    sigma = 1/tau/L**2
    theta = 1.0
    
    E = []

    shape_img = d.shape
    
    u = d 
        
    p= torch.zeros([1,3,shape_img[2], shape_img[3], shape_img[4]]).cuda()


    Dh = torch.tensor([[[0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.]],
                        [[0., 0., 0.],
                        [-1, 1., 0.],
                        [0., 0., 0.]],
                        [[0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
        

    Dp = torch.tensor([[[0., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
    
    Dv = torch.tensor([[[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]],
                    [[0., -1., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
    

    D = torch.concatenate([Dh, Dp, Dv], dim=0)
    
    Dh_star = torch.tensor([[[0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.]],
                        [[0., 0., 0.],
                        [0., 1., -1.],
                        [0., 0., 0.]],
                        [[0., 0., 0.],
                        [0., 0., 0.],
                        [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
        

    Dp_star = torch.tensor([[[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., 1., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
    
    Dv_star = torch.tensor([[[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]],
                    [[0., 0., 0.],
                    [0., 1., 0.],
                    [0., -1., 0.]],
                    [[0., 0., 0.],
                    [0., 0., 0.],
                    [0., 0., 0.]]]).view(1,1,3,3,3).cuda()
    

    D_star = torch.concatenate([Dh_star, Dp_star, Dv_star], dim=1)


    for it in range(0,maxit):

        # remeber old
        u_ = u.clone()

        # primal update               
        u = u- tau*F.conv3d(p,D_star,padding=1)
             
        # proximal step
        u = (u+tau*d)/(1.0+tau)
    
        # overrelaxation
        u_ = u + theta*(u-u_)
        
        # dual update
        p += sigma*F.conv3d(u_,D,padding=1)
        
        # projection
        p = p/torch.clamp(torch.sqrt(torch.sum(torch.mul(p,p),1,True))/lamb,1)
        
        
        if verbose > 0:
            TV1 = lamb * torch.sum(torch.abs(F.conv3d(u,D,padding=1)))
            energy = TV1 + (0.5*torch.sum(torch.mul(u-d,u-d)))
        
    
            E.append(energy)
            if it%check == check-1:
                

                print("iter = ", it,
                      ", tau = ", "{:.3f}".format(tau),
                      ", sigma = ", "{:.3f}".format(sigma),
                      #", time = ", "{:.3f}".format(time.time()-t0),
                      ", E = ", "{:.6f}".format(energy),
                      end="\r")
                
    return u