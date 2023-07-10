import numpy as np
import matplotlib.pyplot as plt
import torch

from .. import wcrg

class NoiseSigmoidWindows(wcrg.Condi_Ansatz):
    """ Scalar potentials given by translated Gaussian windows for white-noised images with std=1"""

    def __init__(self,reconstruct, deconstruct, proj, centers,sigma, device='cpu'):
        
        "Will reconstruct x at the finer scale, without high frequency, and compute the scalar potential of mid freqs conditionaly to low freqs"
        num_potentials = len(centers)
        super().__init__(num_potentials)
        self.device = device #'cpu' or 'cuda'
        self.sigma = sigma #width of the sigmoids
        self.centers = centers #centers of the sigmoids
        self.reconstruct = reconstruct #Reconstruction at finer scale, takes x has mid freqs, x_condi as low freqq and sets high freqs to 0, takes (x_condi, x) (((n_batch,n_pred,Width,Width)),((n_batch,N,N)) returns (n_batch,L,L)
        self.deconstruct = deconstruct
        self.proj = proj
        
        self.points = NoiseSigmoidWindows.prepare_points(self.sigma)
    
    @staticmethod
    def prepare_points(sigma):        
        if torch.is_tensor(sigma):
            min_sigma = sigma.min()
        else:
            min_sigma = sigma
            
        step = max(0.1, min(1, min_sigma) / 4)      
        bound = 4
        n_points = int(2 * bound / step)
        
        return torch.linspace(-bound, bound, 2 * n_points + 1, dtype=torch.float32, device='cuda')
    
    @staticmethod
    def convolve(val, points):
        """
        Parameters:
        val (tensor): phi(x + c) to convolve with p(-c) (*, len(points))
        points (tensor) integration points (len(points))
        
        convolution is done by direct integrarion using Simpson's rule
        
        Returns:
            (tensor) E_c phi(x+c) (*,)
        """
        weights = (2 * np.pi)**(-1/2) * torch.exp(-points**2 / 2)
        f = weights * val
        step = points[1] - points[0]
        return (f[..., 0] + 4 * f[..., 1::2].sum(dim=-1) + 2 * f[..., 2:-1:2].sum(dim=-1) + f[..., -1]) * step / 3
    
    def potential(self, x, x_condi):
        """Potential for single realisation 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_pred,Width,Width)
        x_condi(tensor): Low Frequency x_j (N,N)
        
        Returns:
            (tensor) : sigmoids applied and sumed on each pixel of x_{j-1} (n_potentials,)

        """
        phi = self.reconstruct(x_condi[None], x[None])[0]  # (L,L)
        phi = phi.reshape((-1,))  # (L**2)
        
        # (n_potentials, L**2, len(self.points))
        val = torch.sigmoid(-(phi[None, :, None] + self.points[None, None, :] - self.centers[:, None, None]) / self.sigma[:, None, None])
        return self.convolve(val.sum(dim=1), self.points) #(n_potentials,)

    def potential_batch(self, x, x_condi):
        """Potential for batched realisation 
    
        Parameters:
        x (tensor): High Frequency \bar x_j (n_batch,n_pred,Width,Width)
        x_condi(tensor): Low Frequency x_j (n_batch,N,N)
        
        Returns:
            (tensor) : sigmoids applied and sumed on each pixel of x_{j-1} (n_batch,n_potentials,)

        """
        # (n_batch,n_pred,W,W) and (n_batch,N,N) -> (n_batch,M,)
        phi = self.reconstruct(x_condi, x)  # (batch,L,L)
        phi = phi.reshape(phi.shape[:-2] + (-1,))  # (batch,L**2)
        
        # (n_batch,n_potentials,L**2, len(self.points))
        SIG = torch.sigmoid(-(phi[:, None, :, None] + self.points[None, None, None, :] - self.centers[None, :, None, None]) / (self.sigma[:, None, None] ))
        return self.convolve(SIG.sum(dim=2), self.points) # (n_batch,n_potentials,)
    
    def gradient(self, x, x_condi, theta=None):
        """(N, C, L / 2, L / 2) to (N, M, V, L / 2, L / 2)"""
        
        phi = self.reconstruct(x_condi, x)
        L = phi.shape[-1]
        phi = phi.reshape(phi.shape[:-2] + (-1,))
        
        # (n_batch, n_potentials, L**2, len(self.points))
        sig = torch.sigmoid(-(phi[:, None, :, None] + self.points[None, None, None, :] - self.centers[None, :, None, None]) / (self.sigma[:, None, None]))
        sig_grad = -sig * (1 - sig) / self.sigma[:, None, None]
        sig_grad = self.convolve(sig_grad, self.points).reshape((-1, L, L)) #(n_batch * n_potentials, L, L)
        
        grad_condi, grad_hf = self.deconstruct(sig_grad)
        grad = grad_hf.reshape(sig.shape[:2] + grad_hf.shape[-3:])  
        
        if theta is not None:
            grad = (grad * theta[:, None, None, None]).sum(dim=1)[:, None]
            
        return grad
    
    def laplacian(self, x, x_condi, theta=None):
        """(N, C, L / 2, L / 2) to (N, M)"""
        
        phi = self.reconstruct(x_condi, x)
        L = phi.shape[-1]
        phi = phi.reshape(phi.shape[:-2] + (-1,))
        
        # (n_batch, n_potentials, L**2, len(self.points))
        sig = torch.sigmoid(-(phi[:, None, :, None] + self.points[None, None, None, :] - self.centers[None, :, None, None]) / (self.sigma[:, None, None]))
        sig_lap = sig * (1 - sig) * (1 - 2 * sig) / self.sigma[:, None, None] ** 2
        sig_lap = self.convolve(sig_lap, self.points).reshape((-1, L, L)) #(n_batch * n_potentials, L, L)
        
        lap = torch.sum(sig_lap * self.proj, dim=(1,2))
        lap = lap.reshape(sig.shape[:2])
        
        if theta is not None:
            lap = (lap * theta[:]).sum(1)[:,None]
            
        return lap
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        return (self.laplacian(x,x_condi,theta))
    
class NoiseGaussianPotential(wcrg.Condi_Ansatz):
    """
    For quadratic potentials we employ E_c phi(x+c) = phi(x) + E_c phi(c)
    This is a wrapper around GaussianPotential, which adds this shift
    Laplacian and gradient are unaffected
    """
    def __init__(self, decompose, L, mode = 'All', num_varying_channels=1, num_conditioning_channels=0, shifts=((0, 1), (1, 0))):
        pot = wcrg.GaussianPotential(mode=mode, num_varying_channels=num_varying_channels, 
                                      num_conditioning_channels=num_conditioning_channels, shifts=shifts)
        super().__init__(num_potentials=pot.num_potentials)
        
        self.pot = pot
        self.decompose = decompose
        self.L = L
        self.shift = self.get_shift()
        
    def get_shift(self, batch_size=256):
        sample = torch.randn(batch_size, self.L, self.L, dtype=torch.float32, device='cuda')
        x_condi, x = self.decompose(sample)
        return self.pot.potential_batch(x, x_condi[1]).mean(dim=0)
        
    def potential(self, x, x_condi):
        return self.pot.potential(x, x_condi) + self.shift
    
    def potential_batch(self, x, x_condi):
        return self.pot.potential_batch(x, x_condi) + self.shift
    
    def gradient(self, x, x_condi,theta=None):
        return self.pot.gradient(x, x_condi, theta)
    
    def laplacian(self, x,x_condi,theta=None):
        return self.pot.laplacian(x, x_condi, theta)
    
    def laplacian_Hutchinson(self,x,x_condi,z,theta=None):
        return self.pot.laplacian_Hutchinson(x, x_condi, z, theta)
    
def compute_fast_projection(L, deco, device):
    phi = torch.zeros((L, L, L, L), device=device)
    x, y = torch.meshgrid(torch.arange(L), torch.arange(L), indexing='ij')
    phi[x, y, x, y] = 1

    phi_condi, phi_hf = deco(phi.reshape(-1, L, L))
    proj = (phi_hf ** 2).sum(dim=(1, 2, 3)).reshape((L, L))
    
    return proj

def NoiseANSATZ_Wavelet(W,L,centers,sigma,mode,shifts,shifts_sym = False):
    """Conditionnal ansatz for conditonal Energy \bar E(\bar x_j\vert x{j}) estimation with a wavelet transform,with a scalar potential (sigmoids) + a quadratic potential
    
    Parameters:
    W (Wavelet) : Wavelet to perfom fast wavelet transform
    L (int): system size ( of x_{j-1}) = L*L 
    centers (tensor): position of the centers of the sigmoids 
    sigma (tensor) : width of the sigmoids
    shifts (list of tuples) : spatial shifts for the quadratic potential, carefull (0,0) is already taken into account, do not add here 
    shifts_sym (Bool) : if True, the shifts are not symetrized
    
    
    Returns:
        ansatz_union (Condi_Union) : Ready to be Trained Ansatz
    
    """


  
    reco, deco = wcrg.reconstruct_wav(W), wcrg.decompose_wav(W)
    
    if shifts_sym == False :
        ansatz_gauss = NoiseGaussianPotential(deco, L, mode=mode, num_varying_channels=3,
                                        num_conditioning_channels=1,
                                        shifts = shifts)
    else:
        raise NotImplementedError()
    
    #Scalar potential
    proj = compute_fast_projection(L, deco, device='cuda')
    ansatz_scalar = NoiseSigmoidWindows(reco, deco, proj, centers, sigma, device='cuda')
    #Union
    ansatz_union = wcrg.Condi_Union([ansatz_scalar, ansatz_gauss], condi_index=[0, 1], decompose=deco, reconstruct=reco, device='cuda')
    
    """Dirty"""
    ansatz_union.L = L
    
    return ansatz_union
    
def Plot_NoiseSigmoid(centers,sigma):
    """Show Spatial extent of Sigmoids

    Parameters: 
    centers (tensor): position of the centers of the sigmoids, sorted in increasing order
    sigma (tensor) : width of the sigmoids
    
    
    """
    centers,sigma = centers.cuda(), sigma.cuda()
    window_min = centers[0]
    window_max = centers[-1]
    points = NoiseSigmoidWindows.prepare_points(sigma)
    
    X = torch.linspace(window_min,window_max,1000).cuda()  
    U = torch.sigmoid(-(X[None, :, None] + points[None, None, :] - centers[:, None, None]) / sigma[:,None, None]) 
    U = NoiseSigmoidWindows.convolve(U, points)
    
    plt.plot(X.cpu(), U.transpose(0,1).cpu())
    plt.title('Scalar Potentials')
    plt.ylabel(r'$\rho(t)$')
    plt.xlabel(r'$t$')
    plt.show()
    
def Show_NoiseSigmoid(ansatz_union,add_Trace=True,Free=False,index_scalar=0,index_quad=1) :
    """Shows the learned scalar Potential

    Parameters: 
    ansatz_union (Condi Ansatz): The ansatz 
    add_trace(Bool): if True, we add the trace of the quadratic form
    Free : Whether ansatz_union is a Free Energy (True) or not (False)
    index_scalar (int): position of the quad potential in ansatz_union.ansatze
    index_quad (int): position of the quad potential in ansatz_union.ansatze


    """

    Sc = ansatz_union.ansatze[index_scalar] 
    window_min =Sc.centers[0]
    window_max =Sc.centers[-1]
    
    #POTENTIAL
    X = torch.linspace(window_min,window_max,1000).cuda()  
    U = torch.sigmoid(-(X[None, :, None] + Sc.points[None, None, :] - Sc.centers[:, None, None]) / (Sc.sigma[:,None, None] )) #(n_potentials,n_X, len(Sc.points))
    U = NoiseSigmoidWindows.convolve(U, Sc.points)

    #Num_potentials
    n_pot = 0
    for i in range(0,index_scalar):
        n_pot+=ansatz_union.ansatze[i].num_potentials
    n_scalar = ansatz_union.ansatze[index_scalar].num_potentials
    #theta learned
    theta = ansatz_union.theta()[n_pot:n_pot+n_scalar]

    #Compute Trace
    if add_Trace == True:
        Sq = Square_laplacian(ansatz_union,Free=Free,index_quad=index_quad)
    else:
        Sq =0
    plt.plot(X.cpu().detach(),(theta@U+(Sq/2)*X**2).cpu().detach())
    plt.title('potential')
    plt.show()