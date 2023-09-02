import torch
import numpy as np
from ..scattering import *

class VarBayes:
    def __init__(self, st_calc, threshold_func, mask, images, target_image, eta,
                 noise_size=32, sample_size=5, window=5, blend=0.5):
        self.noise_size = noise_size
        self.sample_size = sample_size
        self.window = window
        
        self.st_calc = st_calc
        self.threshold_func = threshold_func
        self.mask = mask
        
        self.images = [images]
        self.clean_stat = [self.calculate_stat(images)]
        
        self.noisy_stat = []
        self.target = self.threshold_func(self.st_calc.scattering_cov(target_image[None, :, :]), self.mask)[0]
        
        self.eta = eta
        self.alpha, self.beta = None, None
        self.loss_coeff = torch.stack([self.calculate_stat(y + np.random.randn(noise_size, *y.shape)).std(dim=0)**2 for y in images]).mean(dim=0)
        
        self.mu = self.clean_stat[0].mean(dim=0)
        self.sigma = self.clean_stat[0].std(dim=0)  
        
        self.blend = blend
        
    def calculate_stat(self, images):
        s_cov = self.st_calc.scattering_cov(images)
        log_P00 = s_cov['for_synthesis'][:,1:1 + self.st_calc.J * self.st_calc.L]
        stat = self.threshold_func(s_cov, self.mask)
        return torch.cat([log_P00, stat], dim=1)
                        
    def expand_stat(self, s):
        J, L = self.st_calc.J, self.st_calc.L
        log_P00 = s[:, :J * L].reshape((-1, J, L))
        stat = s[:, J * L:]
        return torch.exp(log_P00), stat
        
    def fit(self):
        self.noisy_stat.append(torch.stack([self.calculate_stat(y + np.random.randn(self.noise_size, *y.shape)) for y in self.images[-1]]))
        
        clean_stat_window = torch.cat(self.clean_stat[-self.window:])
        noisy_stat_window = torch.cat(self.noisy_stat[-self.window:])
        
        X = torch.stack([clean_stat_window.T, 
                         torch.ones(clean_stat_window.T.shape, device=clean_stat_window.device)], 
                        dim=-1)
        Y = noisy_stat_window.mean(dim=1).T
        A = torch.linalg.solve((X.mT @ X), torch.sum(X * Y[:, :, None], dim=1))
        
        self.alpha = A[:, 0]
        self.beta = A[:, 1]
        
        n = X.shape[1]
        error = ((torch.sum(X.transpose(0, 1) * A[None, :, :], dim=2) - Y.T) ** 2).sum(dim=0) / (n - 2)
        alpha_error = (error * torch.linalg.inv(X.mT @ X)[:, 0, 0]) ** (1 / 2)
        
        self.sigma = self.blend * (self.eta * self.loss_coeff)**(1/2) / (self.alpha.abs() + alpha_error) + (1 - self.blend) * self.sigma
        
        return (torch.sum(X.transpose(0, 1) * A[None, :, :], dim=2) - Y.T) ** 2
        
    def generate(self):
        eps = torch.randn(self.sample_size, *self.sigma.shape, device=self.sigma.device)
        self.clean_stat.append(self.mu + eps * self.sigma) 
        P00, stat = self.expand_stat(self.clean_stat[-1])
        
        M, N, J, L = self.st_calc.M, self.st_calc.N, self.st_calc.J, self.st_calc.L       
        self.images.append(np.stack([synthesis(
            's_cov_func', mode='estimator', M=M, N=N, J=J, L=L,
            target=stat[i:i+1], reference_P00=P00[i:i+1],
            s_cov_func=self.threshold_func, s_cov_func_params=self.mask)[0] 
                                     for i in range(self.sample_size)]))
