import torch
import torch.nn as nn

class PixelEnergy(nn.Module):
    def __init__(self, centers, sigma):
        super().__init__()
        self.centers = centers
        self.sigma = sigma
        self.theta = torch.nn.Parameter(torch.zeros(len(centers) + 1).to(device), requires_grad=True)
        
    def potential(self, phi, theta=None):
        stat = torch.cat([torch.sigmoid(-(phi[:, None] - self.centers) / self.sigma), phi[:, None] ** 2], dim=1)
        if theta is not None:
            stat = torch.matmul(stat, theta)
        return stat
    
class NoisePixelEnergy(nn.Module):
    def __init__(self, centers, sigma):
        super().__init__()
        self.centers = centers
        self.sigma = sigma
        self.theta = torch.nn.Parameter(torch.zeros(len(centers) + 1).to(device), requires_grad=True)        
        self.points = NoiseSigmoidWindows.prepare_points(self.sigma)
        
    def potential(self, phi, theta=None):
        # (n_batch, n_potentials, len(self.points))
        stat = torch.cat([torch.sigmoid(-(phi[:, None, None] + self.points - self.centers[:, None]) / self.sigma[:, None]), 
                          (phi[:, None, None] + self.points) ** 2], dim=1)
        stat = NoiseSigmoidWindows.convolve(stat, self.points)
        
        if theta is not None:
            stat = torch.matmul(stat, theta)
        return stat
    
def denoise_pixel(ansatz, noise_pixel_ansatz, phi_d, lr=1e-2, n_epochs=100, momentum=0.9, sample_size=65536):
    step = 1
    window_min, window_max = phi_d.min(), phi_d.max()

    pot = ansatz.ansatze[0]
    pixel_ansatz = PixelEnergy(pot.centers, pot.sigma)

    target_stat = pixel_ansatz.potential(phi_d.flatten()).mean(dim=0)
    phi = torch.clamp(phi_d.mean() + phi_d.std() * torch.randn(sample_size, device=device), 
                      min=window_min, max=window_max)

    def closure():
        nonlocal phi, loss_list, iter_num
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
                print('Acceptance rate', np.mean(accept_log))
                accept_log = []  
        
        sample_stat = noise_pixel_ansatz.potential(phi)
        stat = sample_stat.mean(dim=0)
        
        fisher = (sample_stat[:, :, None] * sample_stat[:, None, :]).mean(dim=0) - stat[:, None] * stat[None, :]
        theta.grad = torch.linalg.solve(fisher + 0.01 * torch.eye(len(stat), device=device), target_stat - stat)

        print(theta)
        print(theta.grad)

        loss = (target_stat - sample_stat).abs().mean().item()
        loss_list.append((loss, theta.detach().clone()))
        print('Loss', loss)

        return loss

    loss_list = []
    optimizer = torch.optim.SGD([noise_pixel_ansatz.theta], lr=lr, momentum=momentum)

    for iter_num in range(n_epochs):
        optimizer.step(closure)
        
    return phi, loss_list