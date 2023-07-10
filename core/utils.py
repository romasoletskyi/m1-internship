import numpy as np
import scipy.stats as stats

def preprocessing_euclid(field):
    alpha = 0.5
    epsilon = 1e-6
    
    field_shifted = field - np.min(field) + epsilon
    field_powered = np.power(field_shifted, alpha)
    field_powered = (field_powered - field_powered.mean()) / field_powered.std()

    return field_powered

def preprocessing_weak_lensing(x):
    L = 128
    N = x.shape[1] // L

    x = x.reshape((len(x), N, L, N, L)).swapaxes(2, 3).reshape((len(x) * N * N, L, L))
    x = preprocessing_euclid(x)
    
    return x

def power_spectrum(image):
    assert image.shape[0] == image.shape[1]    
    n = image.shape[0]

    fourier = np.fft.fftn(image)
    amplitude = (np.abs(fourier) ** 2).flatten()

    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten() ** (1 / 2)

    kbins = np.arange(1 / 2, n // 2 + 1, 1)
    kvals = (kbins[1:] + kbins[:-1]) / 2
    bins, _, _ = stats.binned_statistic(knrm, amplitude, statistic = "mean", bins = kbins)
    
    return kvals, bins

def gen_gaussian_field(n, spectrum):
    kfreq = np.fft.fftfreq(n) * n
    kfreq2D = np.meshgrid(kfreq, kfreq)

    knrm = (kfreq2D[0] ** 2 + kfreq2D[1] ** 2).flatten()[None, :] ** (1 / 2)
    kbins = np.arange(1 / 2, n // 2 + 1, 1)[:, None]

    mask = (knrm < kbins[1:]) & (knrm >= kbins[:-1])
    var = np.sum(mask * spectrum[:, None], axis=0) ** (1 / 2)
    fourier_field = var * np.random.randn(n * n) * np.exp(2 * np.pi * 1j * np.random.random(n * n))
    fourier_field = np.reshape(fourier_field, (n, n))
    field = np.fft.ifftn(fourier_field).real
    field = (field - np.mean(field)) / np.std(field)
    
    return field