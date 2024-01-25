import numpy as np
from scipy.special import jv, voigt_profile
import time

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

def my_gaussian(x, mu, sig):
    gauss = np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))*(1/(sig*np.sqrt(2*np.pi)))
    return gauss


def g_distrib_temp_averaged(g,g0,ratio):
    return ((1./np.sqrt(np.pi)*ratio)*(g/g0)**(ratio*ratio))*1./(g*np.sqrt(np.log(g0/g)))


class Pinem:
    def __init__(self, x, amplitude,n_cutoff, kernel = None, rt = 0.7):
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
        self.rt = rt
        self.delta_g = np.linspace(0.0001,1.0-0.0001,1000)
        self.num_g = self.delta_g.shape[0]
        self.amplitude = amplitude
        self.kernel = kernel
        self.n_cutoff = n_cutoff
        self.scale = self.x[1] - self.x[0]
        self.ns, self.fns = self.gen_indices()


    def gen_indices(self):
        tns = np.arange(-self.n_cutoff, self.n_cutoff+1)
        ns = tns[:,np.newaxis]*np.ones((self.num_g,))
        return ns, tns

    def gen_kernel_mattrix(self, omega, offset = 0.0,fwhm= 0.3, seed = 0):
        # To be investigated but I don't think it should change much to use ks or ns.
        fmus = self.fns*omega/self.scale + offset/self.scale
        mus = np.round(fmus).astype(int)
        
        mask_mus = np.abs(mus) < self.x.shape[0]/2
        len_mask = np.sum(mask_mus)
        if self.kernel == 'Voigt' : 
            kernel1D = voigt_fwhm(self.x, fwhm, seed = seed)
        elif self.kernel == 'Gaussian' : 
            kernel1D = my_gaussian(self.x, 0, fwhm/2.355)
        else :
            kernel1D = np.roll(self.kernel)
        kernels = np.repeat(kernel1D[:,np.newaxis], len_mask, axis=1)
        t = np.array([np.roll(kernels[:,i], mus[mask_mus][i]) for i in range(len(mus[mask_mus]))])
        return t, mask_mus

    def calc(self, omega,g,offset=0.0,fwhm = 0.3, seed = 0):
        # mod = np.abs(omega2/omega1 - round_ratio)
        # if mod > self.interference_cutoff:
        #     n_kern_mat, mask_n = self.gen_kernel_mattrix(omega1, ks = False, fwhm = fwhm, seed = seed)
        #     k_kern_mat, mask_k = self.gen_kernel_mattrix(omega2, ks = True, fwhm = fwhm, seed = seed)
        #     j2 = jv(self.ks, 2*g2)[:,np.newaxis][mask_k]*k_kern_mat
        #     j1 = jv(self.ns, 2*g1)[:,np.newaxis][mask_n]*n_kern_mat
        #     wave =self.amplitude*( j2.sum(axis = 0) + j1.sum(axis = 0))
        #     self.kernel_matrix = n_kern_mat
        # else : 
        vg1 = np.tile((g*self.delta_g)[:,np.newaxis],self.ns.shape[0]).T
        g_dist = g_distrib_temp_averaged(vg1, g, self.rt)
        kern_mat, mask_n = self.gen_kernel_mattrix(omega, offset=offset, fwhm = fwhm, seed = seed)
        j = jv(self.ns, 2*vg1)
        js = np.sum((g_dist*j**2)[mask_n,:],axis = 1)[:,np.newaxis]
        wave = self.amplitude*np.sum(kern_mat*js, axis=0)
        self.kernel_matrix = kern_mat

        return wave
    
    def calc_sq_modulus(self, omega, g, offset = 0.0,fwhm = 0.3, seed = 0):
        wave = self.calc(omega,g, offset = offset,fwhm = fwhm, seed = seed)
        return np.real(wave)**2 + np.imag(wave)**2
    
        # omgs = omega1* self.mns + omega2*self.mks
        # allowed = np.where(omgs - omgs.astype(int) ==0,self.mns - self.mks,np.inf)

class PinemGenerator():
    def __init__(
        self,
        n: int,
        scale : float,
        kernel,
        n_cutoff: int = 50,
        amplitude: int = 50,
        rt = 0.7
    ):
        self.n = n

        self._rt = rt
        self.x = np.linspace(-n/2*scale, n/2*scale, n)
        self.kernel = kernel
        self.n_cutoff = n_cutoff
        self._amplitude = amplitude

        self._omg = 1.5
        self._g = None
        self._fwhm = 0.35
        self._offset = None
        self.scale = scale

    @property
    def scale(self) :
        return self._scale
    
    @scale.setter
    def scale(self, value) : 
        self._scale = value
        self.x = np.linspace(-self.n/2*self._scale, self.n/2*self._scale, self.n)
        self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, rt = self._rt)

    @property
    def rt(self) :
        return self._rt
    
    @rt.setter
    def rt(self, value) : 
        self._rt = value
        self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, rt = self._rt)

    @property
    def amplitude (self) :
        return self._amplitude
    
    @amplitude.setter
    def amplitude(self, value) : 
        self._amplitude = value
        self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, rt = self._rt)

    @property
    def omg (self) :
        return self._omg
    
    @omg.setter
    def omg(self, value) :
        self._omg = value
        #self.pinem = Pinem(x = self.x, amplitude = self._amplitude, kernel = self.kernel, n_cutoff = self.n_cutoff, rt = self._rt)
    
    @property
    def g (self) :
        if self._g is None :
            return np.random.uniform(0.1, 2.0)
        else : 
            return self._g
    
    @g.setter
    def g(self, value) :
        self._g = value

    @property
    def offset (self) :
        if self._offset is None :
            return np.random.uniform(-0.5, 0.5)
        else : 
            return self._offset
        
    @offset.setter
    def offset(self, value) :
        self._offset = value

    @property
    def fwhm (self) :
        return self._fwhm
    
    @fwhm.setter
    def fwhm(self, value) :
        self._fwhm = value

    def normalize_spectrum(self, spectrum) :
        M = np.max(spectrum)
        m = np.min(spectrum)
        return (spectrum-m)/(M-m)

    def gen_data(self) : 
        spectre = self.pinem.calc_sq_modulus(omega=self.omg, g =  self.g, offset=self.offset, fwhm = self.fwhm)
        nspectre = self.normalize_spectrum(spectre)*self.amplitude
        noisy_spectre = np.random.poisson(nspectre)
        return noisy_spectre