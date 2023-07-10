import torch

def calculate_shift(clean_ansatz, ansatz, phi):
    L = ansatz.L
    device = phi.device
    
    potential_dim = clean_ansatz.slices[1][0]
    gaussian_dim = clean_ansatz.slices[1][1] - potential_dim
    pot = ansatz.ansatze[0]
    gaussian = clean_ansatz.ansatze[1]

    # gaussian - gaussian

    first_indices = gaussian.first_channel_indices
    second_indices = gaussian.second_channel_indices
    shifts = gaussian.shifts[gaussian.pos_shift_indices]

    delta_cc = first_indices[:, None] == first_indices[None, :]
    delta_dd = second_indices[:, None] == second_indices[None, :]
    delta_ss = torch.all(shifts[:, None, :] == shifts[None, :, :], dim=2)

    delta_cd = first_indices[:, None] == second_indices[None, :]
    delta_dc = delta_cd.T
    delta_sinvs = torch.all(shifts[:, None, :] == -shifts[None, :, :], dim=2)

    delta_hf = (first_indices[None, :] >= gaussian.num_conditioning_channels).float() + (second_indices[None, :] >= gaussian.num_conditioning_channels).float()

    shift_kk = (L / 2)**2 * ((delta_cc * delta_dd * delta_ss).float() + (delta_cd * delta_dc * delta_sinvs).float()) * delta_hf / 4
    shift_kk = shift_kk.cuda()

    # gaussian - potential

    mask = torch.all(shifts == torch.zeros(2), dim=1) * (first_indices == second_indices) * (first_indices >= gaussian.num_conditioning_channels)

    phi_n = torch.randn(phi.shape, device=device)
    sig = torch.sigmoid(-((phi + phi_n)[:, None] - pot.centers) / pot.sigma)
    diff = (0.25 * L**2) * torch.mean(-sig * (1 - sig) * phi_n[:, None], dim=0) / pot.sigma

    shift_ku = torch.zeros(potential_dim, gaussian_dim, device=device)
    shift_ku[:, mask] = diff[:, None]

    # potential - potential

    sig = torch.sigmoid(-((phi + phi_n)[:, None, None] + pot.points - pot.centers[:, None]) / pot.sigma[:, None])
    sig_grad = -sig * (1 - sig) / pot.sigma[:, None]
    sig_grad_conv = pot.convolve(sig_grad, pot.points)

    main = (pot.convolve(sig_grad[:, :, None, :] * sig_grad[:, None, :, :], pot.points)).mean(dim=0)
    mean = (sig_grad_conv[:, :, None] * sig_grad_conv[:, None, :]).mean(dim=0)

    shift_uu = (0.75 * L**2) * (main - mean)

    # assemble

    shift = torch.zeros(potential_dim + gaussian_dim, potential_dim + gaussian_dim, device=device)
    shift[:potential_dim, :potential_dim] = shift_uu
    shift[:potential_dim, potential_dim:] = shift_ku
    shift[potential_dim:, :potential_dim] = shift_ku.T
    shift[potential_dim:, potential_dim:] = shift_kk
    
    return shift

def calculate_rescale(clean_ansatz, dataloader, shift, mean_mode=True):
    grad = []

    for x in dataloader :
        u,v = clean_ansatz.compute_grad(x, None, mean_mode=mean_mode)
        if mean_mode == True:
            u,v = u.cpu()[None],v.cpu()[None]
        else:
            u,v = u.cpu(),v.cpu()
        grad.append((u,v))

    grad_mean = torch.concat([elem[0] for elem in grad],axis=0)
    grad_mean = grad_mean.mean(0).cuda()

    evals, evecs = torch.linalg.eigh(grad_mean - shift)
    pos_evals = torch.clamp(evals, min=0)
    pos_grad = evecs @ torch.diag(pos_evals) @ evecs.T

    return torch.diag(pos_grad)**(-1/2)