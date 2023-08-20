import numpy as np
import torch
import torch.nn as nn

from .noise import NoiseSigmoidWindows

class PixelEnergy(nn.Module):
    def __init__(self, centers, sigma):
        super().__init__()
        self.device = centers.device
        self.centers = centers
        self.sigma = sigma
        self.theta = torch.nn.Parameter(torch.zeros(len(centers) + 1).to(self.device), requires_grad=True)
        
    def potential(self, phi, theta=None):
        stat = torch.cat([torch.sigmoid(-(phi[:, None] - self.centers) / self.sigma), phi[:, None] ** 2], dim=1)
        if theta is not None:
            stat = torch.matmul(stat, theta)
        return stat
    
class NoisePixelEnergy(nn.Module):
    def __init__(self, centers, sigma):
        super().__init__()
        self.device = centers.device
        self.centers = centers
        self.sigma = sigma
        self.theta = torch.nn.Parameter(torch.zeros(len(centers) + 1).to(self.device), requires_grad=True)        
        self.points = NoiseSigmoidWindows.prepare_points(self.sigma)
        
    def potential(self, phi, theta=None):
        # (n_batch, n_potentials, len(self.points))
        stat = torch.cat([torch.sigmoid(-(phi[:, None, None] + self.points - self.centers[:, None]) / self.sigma[:, None]), 
                          (phi[:, None, None] + self.points) ** 2], dim=1)
        stat = NoiseSigmoidWindows.convolve(stat, self.points)
        
        if theta is not None:
            stat = torch.matmul(stat, theta)
        return stat
    
def denoise_pixel(ansatz, noise_pixel_ansatz, phi_d, lr=1e-2, momentum=0.9, n_epochs=100, step=1, sample_size=65536, 
                  verbose=True):
    device = noise_pixel_ansatz.device
    window_min, window_max = phi_d.min(), phi_d.max()

    pot = ansatz.ansatze[0]
    pixel_ansatz = PixelEnergy(pot.centers, pot.sigma)

    target_stat = pixel_ansatz.potential(phi_d.flatten()).mean(dim=0)
    phi = torch.clamp(phi_d.mean() + phi_d.std() * torch.randn(sample_size, device=device), 
                      min=window_min, max=window_max)

    def closure():
        nonlocal phi, loss_list, iter_num
        if verbose:
            print('Iteration', iter_num)
        
        theta = noise_pixel_ansatz.theta
        energy = noise_pixel_ansatz.potential(phi, theta)

        accept_log = []
        for q in range(250):
            new_phi = torch.clamp(phi + step * torch.randn(sample_size, device=device), min=window_min, max=window_max)
            new_energy = noise_pixel_ansatz.potential(new_phi, theta)

            transit_proba = torch.exp(energy - new_energy)
            mask = torch.rand(sample_size, device=device) < transit_proba

            phi = torch.where(mask, new_phi, phi)
            energy = torch.where(mask, new_energy, energy)

            accept_log.append(mask.float().mean().item())
            if (q + 1) % 50 == 0:
                if verbose:
                    print('Acceptance rate', np.mean(accept_log))
                accept_log = []  
        
        sample_stat = noise_pixel_ansatz.potential(phi)
        stat = sample_stat.mean(dim=0)
        
        cov = (sample_stat**2).mean(dim=0) - stat**2
        theta.grad = (target_stat - stat) / cov**(1/2)

        loss = (target_stat - sample_stat).abs().mean().item()
        norm_loss = ((target_stat - sample_stat)**2 / cov).mean().item()
        loss_list.append((loss, norm_loss, theta.detach().clone()))
        
        if verbose:
            print(theta)
            print(theta.grad)
            print('loss', loss, 'norm loss', norm_loss)

        return loss

    loss_list = []
    optimizer = torch.optim.SGD([noise_pixel_ansatz.theta], lr=lr, momentum=momentum)

    for iter_num in range(n_epochs):
        optimizer.step(closure)
        
    return phi, loss_list