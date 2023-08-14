import numpy as np
import torch

def get_triviality_mask(wph_op, true):
    true_stat = wph_op.apply(true, norm='auto')
    mask = np.concatenate([np.ones(len(true_stat), dtype=bool), (true_stat.imag.abs() > 1e-12).cpu().numpy()])
    return mask

def loglikelihood(wph_op, sample, true, epsilon=1e-8):
    stat = wph_op.apply(sample, norm='auto')
    true_stat = wph_op.apply(true, norm='auto')

    mean = torch.cat([stat.real.mean(dim=0), stat.imag.mean(dim=0)])
    std = torch.cat([stat.real.std(dim=0), stat.imag.std(dim=0)]) + epsilon
    val = torch.cat([true_stat.real, true_stat.imag])

    loglike = -torch.log(2 * np.pi * std**2) / 2 - (mean - val)**2 / 2 / std**2
    return loglike.cpu().numpy()    

def variance(wph_op, sample, true, noise_size=256, epsilon=1e-8):
    true_stat = wph_op.apply(true + np.random.randn(noise_size, *true.shape), norm='auto')
    true_std = torch.cat([true_stat.real.std(dim=0), true_stat.imag.std(dim=0)])
    
    sample_mean = torch.stack([wph_op.apply(x + np.random.randn(noise_size, *x.shape), norm='auto').mean(dim=0) for x in sample])
    sample_std = torch.cat([sample_mean.real.std(dim=0), sample_mean.imag.std(dim=0)])
    
    return true_std.cpu().numpy(), sample_std.cpu().numpy()
    
    
def aggregate_metric(wph_op, img, metric, triv_mask):
    """
        wph_op: WPH operator
        img: one sample image to calculate wph object from
        metric: ndarray of metric calculated for each statistic in form (real, imag)
        triv_mask: triviality mask - some imaginary stat are always zero and it is pointless to calculate metrics for them
    """
    
    wph = wph_op(img, ret_wph_obj=True)
    j_index = wph.wph_coeffs_indices[:, 0]

    j_list = np.unique(j_index)
    metric_list = [[], []]

    for j in j_list:
        mask = np.concatenate([j_index == j, np.zeros(len(wph.sm_coeffs), dtype=bool)])
        real_mask = np.concatenate([mask, np.zeros(len(mask), dtype=bool)]) * triv_mask
        imag_mask = np.concatenate([np.zeros(len(mask), dtype=bool), mask]) * triv_mask

        metric_list[0].append(metric[real_mask].mean().item())
        metric_list[1].append(metric[imag_mask].mean().item())

    metric_list = np.array(metric_list)
    return metric_list