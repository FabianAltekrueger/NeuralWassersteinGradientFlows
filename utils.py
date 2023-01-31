# This code belongs to the paper
#
# F. Altekrüger, J. Hertrich and G. Steidl.
# Neural Wasserstein Gradient Flows for Discrepancies with Riesz Kernels
# ArXiv Preprint#2301.11624
#
# Please cite the paper, if you use the code.
# 
# The script provides the code for preparing an image.
#
# The file is an adapted version from 
# 
# H. Wu, J. K{\"o}hler and F. Noe
# Stochastic Normalizing Flows. 
# Advances in Neural Information Processing Systems, volume 33
# (https://github.com/noegroup/stochastic_normalizing_flows)

import numpy as np
from scipy.ndimage import gaussian_filter

def image_to_sample(img, m, gauss_sigma):
    ''' 
    Transforms rgb image into sampled version
    '''
    # make one channel
    img = img.mean(axis=2)
    
    # make background white
    img = img.astype(np.float32)
    img[img > 225] = 255
    
    # normalize
    img /= img.max()

    # convolve with Gaussian
    img2 = gaussian_filter(img, sigma=gauss_sigma)

    # add background
    background1 = gaussian_filter(img, sigma=10)
    background2 = gaussian_filter(img, sigma=20)
    background3 = gaussian_filter(img, sigma=50)
    density = img2 + 0.01 * (background1 + background2 + background3)

    Ix, Iy = np.meshgrid(np.arange(density.shape[1]), np.arange(density.shape[0]))
    idx = np.vstack([Ix.flatten(), Iy.flatten()]).T

    # draw samples from density
    density_normed = density.astype(np.float64)
    density_normed /= density_normed.sum()
    density_flat = density_normed.flatten()
    
    # draw random index
    i = np.random.choice(idx.shape[0], size=m, p=density_flat)
    ixy = idx[i, :]

    # add random noise
    xy = ixy + np.random.rand(m, 2) - 0.5

    # normalize shape
    xy = (xy - np.array([50, 50])) / np.array([100, 100])
    return xy


