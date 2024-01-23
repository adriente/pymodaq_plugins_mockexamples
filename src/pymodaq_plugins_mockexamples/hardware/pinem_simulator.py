import numpy as np
from scipy.special import jv, voigt_profile

# We use this implementation so that the shape is not always the same at the cost of some accuracy on fwhm
def voigt_fwhm(x, fwhm, seed = 0) : 
    if seed == 0:
        seed = np.random.randint(100000)
    np.random.seed(seed)
    ratio = 3.0*np.random.rand(1)
    gamma  = fwhm/(0.5346 + np.sqrt(0.2166 + ratio**2)) #inverting the approximation of https://doi.org/10.1364/JOSA.63.000987
    # it leads to roughly 10% error in the fwhm
    sigma = gamma*ratio
    profile = voigt_profile(x, sigma, gamma)
    return profile/np.max(profile)

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))*(1/(sig*np.sqrt(2*np.pi)))


class Pinem:
    def __init__(self, x, amplitude, k_cutoff, n_cutoff, kernel = None, interference_cutoff = 1e-1):
        """
        x: 1D array of energy values in eV
        kernel: 1D array of the zero-loss peak in the same energy units as x and same length as x
        The kernel is assumed to be centered at x=0
        amplitude: float to scale the intensity of the peaks
        pump: float describing the energy of the subbands in eV
        k_cutoff: int describing the number of subbands to consider
        n_cutoff: int describing the number of subbands to consider
        """
        self.x = x
        self.amplitude = amplitude
        self.kernel = kernel
        self.k_cutoff = k_cutoff
        self.n_cutoff = n_cutoff
        self.scale = self.x[1] - self.x[0]
        self.ns, self.ks, self.mns, self.mks = self.gen_indices()
        self.interference_cutoff = interference_cutoff

    def gen_indices(self):
        ns = np.arange(-self.n_cutoff, self.n_cutoff+1)
        ks = np.arange(-self.k_cutoff, self.k_cutoff+1)
        mns, mks = np.meshgrid(ns, ks)
        return ns, ks, mns, mks

    def gen_kernel_mattrix(self, omega, ks = False ,fwhm= 0.3, seed = 0):
        # To be investigated but I don't think it should change much to use ks or ns.
        if ks :
            mus = (self.ks*omega/self.scale).astype(int)
        else : 
            mus = (self.ns*omega/self.scale).astype(int)
        
        mask_mus = np.abs(mus) < self.x.shape[0]/2
        len_mask = np.sum(mask_mus)
        if self.kernel == 'Voigt' : 
            kernel1D = voigt_fwhm(self.x, fwhm, seed = seed)
        elif self.kernel == 'Gaussian' : 
            kernel1D = gaussian(self.x, 0, fwhm/2.355)
        else :
            kernel1D = self.kernel
        kernels = np.repeat(kernel1D[:,np.newaxis], len_mask, axis=1)
        t = np.array([np.roll(kernels[:,i], mus[mask_mus][i]) for i in range(len(mus[mask_mus]))])
        return t, mask_mus

    def calc(self, omega1, omega2 ,  g1, g2, theta,fwhm = 0.3, seed = 0):
        round_ratio = np.round(omega2/omega1)
        mod = np.abs(omega2/omega1 - round_ratio)
        if mod > self.interference_cutoff:
            n_kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
            k_kern_mat, mask_k = self.gen_kernel_mattrix(omega2, ks = True, fwhm = fwhm, seed = seed)
            j2 = jv(self.ks, 2*g2)[:,np.newaxis][mask_k]*k_kern_mat
            j1 = jv(self.ns, 2*g1)[:,np.newaxis][mask_n]*n_kern_mat
            wave =self.amplitude*( j2.sum(axis = 0) + j1.sum(axis = 0))
            self.kernel_matrix = n_kern_mat
        else : 
            kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
            j2 = (jv(self.ks, 2*g2)*np.exp(1j*self.ks*theta))[:,np.newaxis]
            j1 = jv(self.mns-round_ratio*self.mks, 2*g1)
            js = np.sum(j2*j1, axis=0)[mask_n][:,np.newaxis]
            wave = self.amplitude*np.sum(kern_mat*js, axis=0)
            self.kernel_matrix = kern_mat

        return wave
    
    def calc_sq_modulus(self, omega1, omega2 ,  g1, g2, theta,fwhm = 0.3, seed = 0):
        wave = self.calc(omega1, omega2 ,  g1, g2, theta,fwhm = fwhm, seed = seed)
        return np.real(wave)**2 + np.imag(wave)**2
    
        # omgs = omega1* self.mns + omega2*self.mks
        # allowed = np.where(omgs - omgs.astype(int) ==0,self.mns - self.mks,np.inf)


def gaussian(x, cen, sig):
    return np.exp(-(x-cen)**2/(2*sig**2))

class PinemGenerator():
    def __init__(
        self,
        n: int,
        kernel,
        n_cutoff: int = 50,
        k_cutoff: int = 50,
        amplitude: int = 50
    ):
        self.n = n

        self.kernel = kernel
        self.n_cutoff = n_cutoff
        self.k_cutoff = k_cutoff
        self._amplitude = amplitude
        self.x = None

        self.omg1 = 1.5
        self._omg2 = 2*self.omg1
        self._g1 = None
        self._g2 = None
        self._theta = None
        self._fwhm = 0.35

    @property
    def amplitude (self) :
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, value) : 
        self._amplitude = value
        self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, k_cutoff = self.k_cutoff)

    @property
    def omg1 (self) :
        return self._omg1
    
    @omg1.setter
    def omg1(self, value) :
        self._omg1 = value
        self.x = np.linspace(-30*value, 30*value, self.n)
        self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, k_cutoff = self.k_cutoff)

    @property
    def omg2 (self) :
        return 2*self.omg1
    
    @property
    def g1 (self) :
        if self._g1 is None :
            return np.random.uniform(0.1, 2.0)
        else : 
            return self._g1
    
    @g1.setter
    def g1(self, value) :
        self._g1 = value

    @property
    def g2 (self) :
        if self._g2 is None :
            return np.random.uniform(0.1, 2.0)
        else :
            return self._g2
        
    @g2.setter
    def g2(self, value) :
        self._g2 = value

    @property
    def theta (self) :
        if self._theta is None : 
            return np.random.uniform(0, np.pi)
        else :
            return self._theta
        
    @theta.setter
    def theta(self, value) :
        self._theta = value

    @property
    def fwhm (self) :
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value) :
        self._fwhm = value

    def gen_data(self) : 
        spectre = self.pinem.calc_sq_modulus(omega1=self.omg1,omega2 = self.omg2, g1 =  self.g1, g2 = self.g2, theta =  self.theta, fwhm = self.fwhm)
        noisy_spectre = np.random.poisson(spectre)
        return noisy_spectre