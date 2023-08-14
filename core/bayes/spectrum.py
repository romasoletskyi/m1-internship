import numpy as np

from ..utils import power_spectrum, gen_gaussian_field

def sample_gaussian_init_cond(reconstruct_img, noise_img, slope_list):
    y0_list = []
    
    noise_k, noise_bins = power_spectrum(noise_img)
    k, bins = power_spectrum(reconstruct_img)

    for slope in slope_list:
        clean_mask = noise_bins < bins
        k_clean = k[clean_mask]
        k_noisy = k[~clean_mask]

        bins_clean = bins[clean_mask]
        bins_gen = np.concatenate([bins_clean, bins_clean[-1] * (k_noisy / k_clean[-1]) ** slope])

        y0 = gen_gaussian_field(reconstruct_img.shape[0], bins_gen)
        y0_list.append(y0)
     
    y0_list = np.stack(y0_list)
    return y0_list
