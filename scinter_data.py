import numpy as np
from numpy import newaxis as na
import os
from numpy.ctypeslib import ndpointer
import ctypes
import progressbar
import scipy
import scipy.interpolate as interp
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import skimage
from skimage.measure import block_reduce

dir_path = os.path.dirname(os.path.realpath(__file__))
file_c = os.path.join(os.path.join(dir_path,"libcpp"),"lib_scinter.so")
lib = ctypes.CDLL(file_c)

#constants
au = 149597870700. #m
pc = 648000./np.pi*au #m
day = 24*3600
year = 365.2425*day
degrees = np.pi/180.
mas = degrees/1000./3600.
v_c = 299792458. #m/s
MHz = 1.0e+6
mHz = 1.0e-3
musec = 1.0e-6
mus = musec
sqrtmus = np.sqrt(musec)
e = 1.602176634e-19 #C
me = 9.1093837015e-31 #kg
eps0 = 8.8541878128e-12 #SI
pc_per_cm3 = pc/0.01**3
hour = 3600.
minute = 60.
kms = 1000.

#load C++ library for fast NuT transform
lib.NuT.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_fD
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tt [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fD [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_im [N_t*N_nu]
] 

#load C++ library for computation of a secondary spectrum from stable gradients
lib.SSgrad.argtypes = [
    ctypes.c_int,   # N_t = N_fD
    ctypes.c_int,   # N_nu = N_tau
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tt [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fD [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tau [N_nu]
    ctypes.c_double,   # nu0
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # SS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # SS_im [N_t*N_nu]
] 

#load C++ library for fast NuT transform including correction of refraction
lib.NuT_derefracted.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_fD
    ctypes.c_double,   # nu0
    ctypes.c_double,   # f_refr
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tt [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fD [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_im [N_t*N_nu]
]

#load C++ library for fast Fourier detection of 1/nu^2 refraction
lib.SS_refr.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_fnum2
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # num2 [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fnum2 [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_im [N_t*N_nu]
] 

lib.ENuT.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_fD
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tt [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # fD [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hWF_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hWF_im [N_t*N_nu]
] 

lib.Lambda.argtypes = [
    ctypes.c_int,   # N_t
    ctypes.c_int,   # N_nu
    ctypes.c_int,   # N_tau
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # L [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # tau [N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DS [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_real [N_t*N_nu]
    ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # hSS_im [N_t*N_nu]
]  

class intensity:
    type = "intensity"
    
    def __init__(self,data_path):
        """
        override this function to load custom data
        self.DS: dynamic spectrum time*frequency
        self.nu: frequency
        self.t: time in seconds since start of observation
        self.mjd: time in MJD
        """
        # - load data
        self.data_path = data_path
        file_data = os.path.join(data_path,"DS.npz")
        lib_data = np.load(file_data)
        self.nu = lib_data["nu"]
        self.t = lib_data["t"]
        self.mjd = lib_data["mjd"]
        self.DS = lib_data["DS"]
        # - provide some useful parameters
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.DS.shape
        self.dt = np.mean(np.diff(self.t)) #self.t[1]-self.t[0]
        self.dnu = np.mean(np.diff(self.nu)) #self.nu[1]-self.nu[0]
        self.dmjd = np.mean(np.diff(self.mjd))
        self.t_min = self.t[0]
        self.t_max = self.t[-1]
        self.nu_min = self.nu[0]
        self.nu_max = self.nu[-1]
        self.timespan = self.t_max-self.t_min
        self.bandwidth = self.nu_max-self.nu_min
        self.t0 = np.mean(self.t)
        self.nu0 = np.mean(self.nu)
        self.mjd_min = self.mjd[0]
        self.mjd_max = self.mjd[-1]
        self.mjd0 = np.mean(self.mjd)
        
    def crop(self,t_min=None,t_max=None,nu_min=None,nu_max=None,N_t=None,N_nu=None): #missing option for mask, profile and bpass
            # - create subset of data
        if t_min==None:
            i0_t = 0
        else:
            i0_t = np.argmin(np.abs(self.t-t_min))
        if t_max==None:
            i1_t = self.N_t
            if not N_t==None:
                if N_t <= i1_t:
                    i1_t = N_t
                else:
                    print("/!\ N_t too large! Using available data instead.")
        else:
            i1_t = np.argmin(np.abs(self.t-t_max))
            if not N_t==None and N_t!=i1_t:
                print("/!\ N_t incompatible with t_max! Using only t_max instead.")
        if nu_min==None:
            i0_nu = 0
        else:
            i0_nu = np.argmin(np.abs(self.nu-nu_min))
        if nu_max==None:
            i1_nu = self.N_nu
            if not N_nu==None:
                if N_nu <= i1_nu:
                    i1_nu = N_nu
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_nu = np.argmin(np.abs(self.nu-nu_max))
            if not N_nu==None and N_nu!=i1_nu:
                print("/!\ N_nu incompatible with nu_max! Using only nu_max instead.")
        if i0_t!=0 or i1_t!=self.N_t or i0_nu!=0 or i1_nu!=self.N_nu:
            self.mjd = self.mjd[i0_t:i1_t]
            self.t = self.t[i0_t:i1_t]
            self.nu = self.nu[i0_nu:i1_nu]
            self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
            self.recalculate()
            
    def slice(self,i0_t=0,i1_t=-1,i0_nu=0,i1_nu=-1):
        self.mjd = self.mjd[i0_t:i1_t]
        self.t = self.t[i0_t:i1_t]
        self.nu = self.nu[i0_nu:i1_nu]
        self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
        self.recalculate()
            
    def downsample(self,**kwargs):
        t_sampling = kwargs.get("t_sampling",1)
        nu_sampling = kwargs.get("nu_sampling",1)
        self.DS = block_reduce(self.DS, block_size=(t_sampling,nu_sampling), func=np.mean)
        coordinates = np.array([self.t,self.t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=self.t[-1])
        self.t = coordinates[0,:]
        coordinates = np.array([self.mjd,self.mjd])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=self.mjd[-1])
        self.mjd = coordinates[0,:]
        coordinates = np.array([self.nu,self.nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=self.nu[-1])
        self.nu = coordinates[0,:]
        self.recalculate()
        
    def slice_zeta(self,**kwargs):
        tchunk = kwargs.get("tchunk",60)
        nuchunk = kwargs.get("nuchunk",200)
        N_th = kwargs.get("N_th",513)
        npad = kwargs.get("npad",3)
        stau_max = kwargs.get("stau_max",20.e-06)
        zeta_min = kwargs.get("zeta_min",1.)
        zeta_max = kwargs.get("zeta_max",2.)
        N_zeta = kwargs.get("N_zeta",120)
        fit_radius = kwargs.get("fit_radius",1.)
        diagnostics = kwargs.get("diagnostics",False)
        experimental = kwargs.get("experimental",False)
        
        hnuchunk = int(nuchunk/2)
        htchunk = int(tchunk/2)
        N_tchunk = self.N_t//tchunk
        N_nuchunk = (self.N_nu-hnuchunk)//hnuchunk
        chunks = np.zeros((N_tchunk,N_nuchunk,tchunk,nuchunk),dtype=complex)
        staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
        E_recov = np.zeros((self.N_t,self.N_nu),dtype=complex)
        
        def chi_par(x, A, x0, C):
            """
            Parabola for fitting to chisq curve.
            """
            return A*(x - x0)**2 + C
        
        t_var = np.zeros(N_tchunk)
        zeta_var = np.zeros(N_tchunk)
        zeta_err_var = np.zeros(N_tchunk)
        # flags = []
        if experimental:
            eig_cum = np.zeros((N_tchunk,N_zeta),dtype=float)
        
        for tc in range(N_tchunk):
            f0 = []
            zeta_evo = []
            zeta_evo_err = []
            zeta_fit0=0.
            zeta_sig0=0.
            
            t_chunk = self.t[tc*tchunk:(tc+1)*tchunk]
            t_var[tc] = t_chunk.mean()
            
            for fc in range(N_nuchunk):
                failed=False
                dspec = self.DS[tc*tchunk:(tc+1)*tchunk,fc*nuchunk:(fc+1)*nuchunk]
                dspec = dspec - dspec.mean()
                nu_chunk = self.nu[fc*nuchunk:(fc+1)*nuchunk]
                f0_evo = nu_chunk.mean()
                dspec_pad = np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
                
                if not len(nu_chunk)*len(t_chunk)==0:
                    ##Form SS and coordinate arrays
                    SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                    fd = np.fft.fftshift(np.fft.fftfreq((npad+1)*t_chunk.shape[0],t_chunk[1]-t_chunk[0]))
                    tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nu_chunk.shape[0],nu_chunk[1]-nu_chunk[0]))
                
                    ##Setup for chisq search
                    zetas = np.linspace(zeta_min,zeta_max,num=N_zeta)
                    eigs = np.zeros(N_zeta)
            
                    ##Determine chisq for each delay drift
                    for i in range(N_zeta):
                        zeta = zetas[i]
                        stau_to_fD = 2.*f0_evo*zeta
                        #print(i,zeta)
                        ##Fits a model generated by taking the outer product of the dominate eigenvector
                        ##(in theta-theta space) and then mapping back to the dynamic spectrum
                        #Compute thth diagram
                        th1 = np.ones((N_th,N_th))*staus
                        th2 = th1.T
                        dfd = np.diff(fd).mean()
                        dtau = np.diff(tau).mean()
                        tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                        fd_inv = ((stau_to_fD*(th1-th2)-fd[0]+dfd/2)//dfd).astype(int)
                        thth = np.zeros((N_th,N_th), dtype=complex)
                        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv > 0) * (fd_inv < fd.shape[0])
                        thth[pnts] = SS[fd_inv[pnts],tau_inv[pnts]]
                        eta = 1./(2.*f0_evo*zeta)**2
                        thth *= np.sqrt(np.abs(2*eta*stau_to_fD*(th2-th1))) #flux conervation
                        thth /= np.mean(np.abs(thth))
                        if 1:
                            thth -= np.tril(thth) #make hermitian
                            thth += np.conjugate(np.triu(thth).T)
                            thth -= np.diag(np.diag(thth))
                            thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                            thth = np.nan_to_num(thth)
                        else:
                            #Produces a similar but slightly different result
                            thth = (thth+np.conj(np.transpose(thth)))/2. #assert hermitian
                        ##Find first eigenvector and value
                        v0 = thth[thth.shape[0]//2,:]
                        v0 /= np.sqrt((np.abs(v0)**2).sum())
                        w,V = eigsh(thth,1,v0=v0,which='LA')
                        eigs[i] = np.abs(w[0])
            
                        if experimental and np.all(dspec!=0):
                            eig_cum[tc,i] += eigs[i]
                    if experimental:
                        print(t_var[tc]/60.,f0_evo/1.0e+6)
                    else:
                        #Fit for a parabola around the maximum
                        try:
                            z_max = zetas[eigs==eigs.max()][0]
                            zetas_fit = zetas[np.abs(zetas-z_max)<fit_radius]
                            eigs_fit = eigs[np.abs(zetas-z_max)<fit_radius]
                            C = eigs_fit.max()
                            x0 = zetas_fit[eigs_fit==C][0]
                            A = (eigs_fit[0]-C)/((zetas_fit[0]-x0)**2)
                            popt,pcov = curve_fit(chi_par,zetas_fit,eigs_fit,p0=np.array([A,x0,C]))
                            zeta_fit = popt[1]
                            zeta_sig = np.sqrt(-(eigs_fit-chi_par(zetas_fit,*popt)).std()/popt[0])
                            if np.isnan(zeta_sig) or zeta_sig==0.0 or (z_max in [zetas[0],zetas[1],zetas[3],zetas[-1],zetas[-2],zetas[-3]]):
                                print('invalid fit')
                                zeta_fit = zeta_fit0
                                zeta_sig = zeta_sig0*3.
                                failed=True
                        except:
                            print('failed')
                            zeta_fit = zeta_fit0
                            zeta_sig = zeta_sig0*3.
                            failed = True
                        if not failed:
                            zeta_evo.append(zeta_fit)
                            zeta_evo_err.append(zeta_sig)
                            f0.append(f0_evo)
                            zeta_fit0 = zeta_fit
                            zeta_sig0 = zeta_sig
                            eta = 1./(2.*f0_evo*zeta_fit)**2
                            print(t_var[tc]/60.,f0_evo/1.0e+6,zeta_fit/sqrtmus*hour,zeta_sig/sqrtmus*hour,eta)
                        if diagnostics:
                            path_diagnostics = os.path.join(self.data_path,"diagnostics_slice_zeta")
                            if not os.path.exists(path_diagnostics):
                                os.mkdir(path_diagnostics)
                            figure = plt.figure(figsize=(18,7))
                            plt.plot(zetas/sqrtmus*hour,eigs,markersize=2,linestyle="",marker=".")
                            plt.xlabel(r"delay drift [$\sqrt{\mu s}$/h]")
                            plt.ylabel("eigenvalue")
                            figure.savefig(os.path.join(path_diagnostics,"{0}_{1}.png".format(tc,fc)))
                            plt.close()
            
            if experimental:
                zeta_var[tc] = zetas[np.argmax(eig_cum[tc,:])]
                zeta_err_var[tc] = zetas[2]-zetas[1]
                if diagnostics:
                    path_diagnostics = os.path.join(self.data_path,"diagnostics_slice_zeta")
                    if not os.path.exists(path_diagnostics):
                        os.mkdir(path_diagnostics)
                    figure = plt.figure(figsize=(18,7))
                    plt.plot(zetas/sqrtmus*hour,eig_cum[tc,:],markersize=2,linestyle="",marker=".")
                    plt.xlabel(r"delay drift [$\sqrt{\mu s}$/h]")
                    plt.ylabel("summed eigenvalue")
                    figure.savefig(os.path.join(path_diagnostics,"{0}.png".format(tc)))
                    plt.close()
            else:
                zeta_evo = np.array(zeta_evo)
                zeta_evo_err = np.array(zeta_evo_err)
                f0 = np.array(f0)
                
                #throw away last empty slice
                f0 = f0[:-1]
                zeta_evo = zeta_evo[:-1]
                zeta_evo_err = zeta_evo_err[:-1]
                
                zeta_var[tc] = np.mean(zeta_evo)
                zeta_err_var[tc] = np.std(zeta_evo)/np.sqrt(len(zeta_evo))
                
                
        return t_var,zeta_var,zeta_err_var
        
    def slice_eta(self,**kwargs):
        tchunk = kwargs.get("tchunk",60)
        nuchunk = kwargs.get("nuchunk",200)
        N_th = kwargs.get("N_th",513)
        npad = kwargs.get("npad",3)
        stau_max = kwargs.get("stau_max",20.e-06)
        eta_min = kwargs.get("eta_min",1.)
        eta_max = kwargs.get("eta_max",2.)
        N_eta = kwargs.get("N_eta",120)
        fit_radius = kwargs.get("fit_radius",1.)
        diagnostics = kwargs.get("diagnostics",False)
        
        hnuchunk = int(nuchunk/2)
        htchunk = int(tchunk/2)
        N_tchunk = self.N_t//tchunk
        N_nuchunk = (self.N_nu-hnuchunk)//hnuchunk
        chunks = np.zeros((N_tchunk,N_nuchunk,tchunk,nuchunk),dtype=complex)
        staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
        E_recov = np.zeros((self.N_t,self.N_nu),dtype=complex)
        
        def chi_par(x, A, x0, C):
            """
            Parabola for fitting to chisq curve.
            """
            return A*(x - x0)**2 + C
        
        t_var = np.zeros(N_tchunk)
        eta_var = np.zeros(N_tchunk)
        eta_err_var = np.zeros(N_tchunk)
        # flags = []
        nu_var = np.zeros(N_nuchunk)
        etas_arr = np.zeros((N_tchunk,N_nuchunk))
        errs_arr = np.zeros((N_tchunk,N_nuchunk))
        
        for tc in range(N_tchunk):
            f0 = []
            eta_evo = []
            eta_evo_err = []
            eta_fit0=0.
            eta_sig0=0.
            
            t_chunk = self.t[tc*htchunk:(tc+1)*htchunk]
            t_var[tc] = t_chunk.mean()
            
            for fc in range(N_nuchunk):
                failed=False
                dspec = self.DS[tc*htchunk:(tc+1)*htchunk,fc*hnuchunk:(fc+1)*hnuchunk]
                dspec = dspec - dspec.mean()
                nu_chunk = self.nu[fc*hnuchunk:(fc+1)*hnuchunk]
                f0_evo = nu_chunk.mean()
                nu_var[fc] = f0_evo
                dspec_pad = np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
                
                if not len(nu_chunk)*len(t_chunk)==0:
                    ##Form SS and coordinate arrays
                    SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                    fd = np.fft.fftshift(np.fft.fftfreq((npad+1)*t_chunk.shape[0],t_chunk[1]-t_chunk[0]))
                    tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nu_chunk.shape[0],nu_chunk[1]-nu_chunk[0]))
                
                    ##Setup for chisq search
                    etas = np.linspace(eta_min,eta_max,num=N_eta)
                    eigs = np.zeros(N_eta)
            
                    ##Determine chisq for each delay drift
                    for i in range(N_eta):
                        eta = etas[i]
                        stau_to_fD = 1./np.sqrt(eta)
                        #print(i,eta)
                        ##Fits a model generated by taking the outer product of the dominate eigenvector
                        ##(in theta-theta space) and then mapping back to the dynamic spectrum
                        #Compute thth diagram
                        th1 = np.ones((N_th,N_th))*staus
                        th2 = th1.T
                        dfd = np.diff(fd).mean()
                        dtau = np.diff(tau).mean()
                        tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                        fd_inv = ((stau_to_fD*(th1-th2)-fd[0]+dfd/2)//dfd).astype(int)
                        thth = np.zeros((N_th,N_th), dtype=complex)
                        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv > 0) * (fd_inv < fd.shape[0])
                        thth[pnts] = SS[fd_inv[pnts],tau_inv[pnts]]
                        thth *= np.sqrt(np.abs(2*eta*stau_to_fD*(th2-th1))) #flux conervation
                        thth /= np.mean(np.abs(thth))
                        if 1:
                            thth -= np.tril(thth) #make hermitian
                            thth += np.conjugate(np.triu(thth).T)
                            thth -= np.diag(np.diag(thth))
                            thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                            thth = np.nan_to_num(thth)
                        else:
                            #Produces a similar but slightly different result
                            thth = (thth+np.conj(np.transpose(thth)))/2. #assert hermitian
                        ##Find first eigenvector and value
                        v0 = thth[thth.shape[0]//2,:]
                        v0 /= np.sqrt((np.abs(v0)**2).sum())
                        w,V = eigsh(thth,1,v0=v0,which='LA')
                        eigs[i] = np.abs(w[0])
            
                    #Fit for a parabola around the maximum
                    try:
                        e_max = etas[eigs==eigs.max()][0]
                        fit_radius0 = 0.
                        if fit_radius==None:
                            fit_radius0 = .1*e_max
                        else:
                            fit_radius0 = fit_radius
                        etas_fit = etas[np.abs(etas-e_max)<fit_radius0]
                        eigs_fit = eigs[np.abs(etas-e_max)<fit_radius0]
                        C = eigs_fit.max()
                        x0 = etas_fit[eigs_fit==C][0]
                        A = (eigs_fit[0]-C)/((etas_fit[0]-x0)**2)
                        popt,pcov = curve_fit(chi_par,etas_fit,eigs_fit,p0=np.array([A,x0,C]))
                        eta_fit = popt[1]
                        eta_sig = np.sqrt(-(eigs_fit-chi_par(etas_fit,*popt)).std()/popt[0])
                        if np.isnan(eta_sig) or eta_sig==0.0 or (e_max in [etas[0],etas[1],etas[3],etas[-1],etas[-2],etas[-3]]):
                            print('invalid fit')
                            eta_fit = eta_fit0
                            eta_sig = eta_sig0*3.
                            failed=True
                    except:
                        print('failed')
                        eta_fit = eta_fit0
                        eta_sig = eta_sig0*3.
                        failed = True
                    if not failed:
                        eta_evo.append(eta_fit)
                        eta_evo_err.append(eta_sig)
                        f0.append(f0_evo)
                        eta_fit0 = eta_fit
                        eta_sig0 = eta_sig
                        zeta = 1./(2.*f0_evo*np.sqrt(eta_fit))/sqrtmus*hour
                        print(t_var[tc]/60.,f0_evo/1.0e+6,eta_fit,eta_sig,zeta)
                        etas_arr[tc,fc] = eta_fit
                        errs_arr[tc,fc] = eta_sig
                    else:
                        etas_arr[tc,fc] = np.nan
                        errs_arr[tc,fc] = np.nan
                    if diagnostics:
                        path_diagnostics = os.path.join(self.data_path,"diagnostics_slice_eta")
                        if not os.path.exists(path_diagnostics):
                            os.mkdir(path_diagnostics)
                        figure = plt.figure(figsize=(18,7))
                        plt.plot(etas,eigs,markersize=2,linestyle="",marker=".")
                        plt.xlabel(r"curvature [s$^3$]")
                        plt.ylabel("eigenvalue")
                        if not failed:
                            plt.plot(etas_fit,chi_par(etas_fit,*popt),markersize=0,linestyle='-',color='red')
                        figure.savefig(os.path.join(path_diagnostics,"{0}_{1}.png".format(tc,fc)))
                        plt.close()
                        
            eta_evo = np.array(eta_evo)
            eta_evo_err = np.array(eta_evo_err)
            f0 = np.array(f0)
            
            # #throw away last empty slice
            # f0 = f0[:-1]
            # eta_evo = eta_evo[:-1]
            # eta_evo_err = eta_evo_err[:-1]
            
            if diagnostics:
                outfile = os.path.join(path_diagnostics,"{0}.png".format(tc))
                figure = plt.figure(figsize=(18,7))
                plt.plot(etas,eigs,markersize=2,linestyle="",marker=".")
                plt.ylabel("freq [MHz]")
                plt.ylabel(r"curvature [s$^3$]")
                plt.xlim([f0[0]/MHz,f0[-1]/MHz])
                plt.ylim([eta_min,eta_max])
                if not failed:
                    plt.errorbar(f0/MHz,eta_evo,yerr=eta_evo_err,markersize=5,linestyle='',color='red')
                figure.savefig(outfile)
                plt.close()
                #np.savez(outfile,eta_evo=eta_evo,eta_evo_err=eta_evo_err,f0=f0)
            
            eta1400_evo = eta_evo*f0**2/(1400.*MHz)**2
            eta_var[tc] = np.mean(eta1400_evo)
            eta_err_var[tc] = np.std(eta1400_evo)/np.sqrt(len(eta1400_evo))
                
        return t_var,eta_var,eta_err_var,nu_var,etas_arr,errs_arr
    
class dynspec_masked(intensity):
    def __init__(self,data_path):
        """
        override this function to load custom data
        self.DS: dynamic spectrum time*frequency
        self.nu: frequency
        self.t: time in seconds since start of observation
        self.mjd: time in MJD
        """
        # - load data
        self.data_path = data_path
        file_data = os.path.join(data_path,"DS.npz")
        lib_data = np.load(file_data)
        self.nu = lib_data["nu"]
        self.t = lib_data["t"]
        self.mjd = lib_data["mjd"]
        self.DS = lib_data["DS"]
        self.mask = lib_data["mask"]
        # - provide some useful parameters
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.DS.shape
        self.dt = np.mean(np.diff(self.t)) #self.t[1]-self.t[0]
        self.dnu = np.mean(np.diff(self.nu)) #self.nu[1]-self.nu[0]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]
        self.nu_min = self.nu[0]
        self.nu_max = self.nu[-1]
        self.timespan = self.t_max-self.t_min
        self.bandwidth = self.nu_max-self.nu_min
        self.t0 = np.mean(self.t)
        self.nu0 = np.mean(self.nu)
        self.mjd_min = self.mjd[0]
        self.mjd_max = self.mjd[-1]
        self.mjd0 = np.mean(self.mjd)
        
    def crop(self,t_min=None,t_max=None,nu_min=None,nu_max=None,N_t=None,N_nu=None): #missing option for mask, profile and bpass
            # - create subset of data
        if t_min==None:
            i0_t = 0
        else:
            i0_t = np.argmin(np.abs(self.t-t_min))
        if t_max==None:
            i1_t = self.N_t
            if not N_t==None:
                if N_t <= i1_t:
                    i1_t = N_t
                else:
                    print("/!\ N_t too large! Using available data instead.")
        else:
            i1_t = np.argmin(np.abs(self.t-t_max))
            if not N_t==None and N_t!=i1_t:
                print("/!\ N_t incompatible with t_max! Using only t_max instead.")
        if nu_min==None:
            i0_nu = 0
        else:
            i0_nu = np.argmin(np.abs(self.nu-nu_min))
        if nu_max==None:
            i1_nu = self.N_nu
            if not N_nu==None:
                if N_nu <= i1_nu:
                    i1_nu = N_nu
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_nu = np.argmin(np.abs(self.nu-nu_max))
            if not N_nu==None and N_nu!=i1_nu:
                print("/!\ N_nu incompatible with nu_max! Using only nu_max instead.")
        if i0_t!=0 or i1_t!=self.N_t or i0_nu!=0 or i1_nu!=self.N_nu:
            self.mjd = self.mjd[i0_t:i1_t]
            self.t = self.t[i0_t:i1_t]
            self.nu = self.nu[i0_nu:i1_nu]
            self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
            self.mask = self.mask[i0_t:i1_t,i0_nu:i1_nu]
            self.recalculate()
        else:
            raise ValueError
            
    def slice(self,i0_t=0,i1_t=-1,i0_nu=0,i1_nu=-1):
        self.mjd = self.mjd[i0_t:i1_t]
        self.t = self.t[i0_t:i1_t]
        self.nu = self.nu[i0_nu:i1_nu]
        self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
        self.mask = self.mask[i0_t:i1_t,i0_nu:i1_nu]
        self.recalculate()
            
    def downsample(self,**kwargs):
        t_sampling = kwargs.get("t_sampling",1)
        nu_sampling = kwargs.get("nu_sampling",1)
        masked_DS = np.copy(self.DS)
        masked_DS[self.mask] = np.nan
        filler = block_reduce(self.DS, block_size=(t_sampling,nu_sampling), func=np.mean)
        self.DS = block_reduce(masked_DS, block_size=(t_sampling,nu_sampling), func=np.nanmean)
        self.mask = block_reduce(self.mask, block_size=(t_sampling,nu_sampling), func=np.min)
        self.DS[self.mask] = filler[self.mask]
        coordinates = np.array([self.t,self.t])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=self.t[-1])
        self.t = coordinates[0,:]
        coordinates = np.array([self.mjd,self.mjd])
        coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=self.mjd[-1])
        self.mjd = coordinates[0,:]
        coordinates = np.array([self.nu,self.nu])
        coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=self.nu[-1])
        self.nu = coordinates[0,:]
        self.recalculate()
    
        
class visibility:
    type = "visibility"
    
    def __init__(self,data_path):
        """
        override this function to load custom data
        self.DS: dynamic spectrum time*frequency
        self.nu: frequency
        self.t: time in seconds since start of observation
        self.mjd: time in MJD
        """
        # - load data
        self.data_path = data_path
        file_data = os.path.join(data_path,"DS.npz")
        lib_data = np.load(file_data)
        self.nu = lib_data["nu"]
        self.t = lib_data["t"]
        self.mjd = lib_data["mjd"]
        self.DS = lib_data["DS"]
        self.phase = lib_data["phase"]
        # - provide some useful parameters
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.DS.shape
        self.dt = self.t[1]-self.t[0]
        self.dnu = self.nu[1]-self.nu[0]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]
        self.nu_min = self.nu[0]
        self.nu_max = self.nu[-1]
        self.timespan = self.t_max-self.t_min
        self.bandwidth = self.nu_max-self.nu_min
        self.t0 = np.mean(self.t)
        self.nu0 = np.mean(self.nu)
        self.mjd_min = self.mjd[0]
        self.mjd_max = self.mjd[-1]
        self.mjd0 = np.mean(self.mjd)
        
    def crop(self,t_min=None,t_max=None,nu_min=None,nu_max=None,N_t=None,N_nu=None): #missing option for mask, profile and bpass
            # - create subset of data
        if t_min==None:
            i0_t = 0
        else:
            i0_t = np.argmin(np.abs(self.t-t_min))
        if t_max==None:
            i1_t = self.N_t
            if not N_t==None:
                if N_t <= i1_t:
                    i1_t = N_t
                else:
                    print("/!\ N_t too large! Using available data instead.")
        else:
            i1_t = np.argmin(np.abs(self.t-t_max))
            if not N_t==None and N_t!=i1_t:
                print("/!\ N_t incompatible with t_max! Using only t_max instead.")
        if nu_min==None:
            i0_nu = 0
        else:
            i0_nu = np.argmin(np.abs(self.nu-nu_min))
        if nu_max==None:
            i1_nu = self.N_nu
            if not N_nu==None:
                if N_nu <= i1_nu:
                    i1_nu = N_nu
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_nu = np.argmin(np.abs(self.nu-nu_max))
            if not N_nu==None and N_nu!=i1_nu:
                print("/!\ N_nu incompatible with nu_max! Using only nu_max instead.")
        if i0_t!=0 or i1_t!=self.N_t or i0_nu!=0 or i1_nu!=self.N_nu:
            self.mjd = self.mjd[i0_t:i1_t]
            self.t = self.t[i0_t:i1_t]
            self.nu = self.nu[i0_nu:i1_nu]
            self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
            self.phase = self.phase[i0_t:i1_t,i0_nu:i1_nu]
            self.recalculate()
            
    def slice(self,i0_t=0,i1_t=-1,i0_nu=0,i1_nu=-1):
        self.mjd = self.mjd[i0_t:i1_t]
        self.t = self.t[i0_t:i1_t]
        self.nu = self.nu[i0_nu:i1_nu]
        self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
        self.phase = self.phase[i0_t:i1_t,i0_nu:i1_nu]
        self.recalculate()
        
class secondary_spectrum:
    type = "secondary spectrum"
    
    def __init__(self):
        """
        override this
        self.SS: secondary spectrum (|I|^2) Doppler*delay
        self.fD: Doppler rate
        self.tau: delay
        """
        pass
        
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_tau = self.SS.shape
        self.dfD = self.fD[1] - self.fD[0]
        self.dtau = self.tau[1] - self.tau[0]
        self.fD_max = np.max(self.fD)
        self.tau_max = np.max(self.tau)
        
class conjugate_spectrum:
    type = "conjugate spectrum"
    
    def __init__(self):
        """
        override this
        self.CS: conjugate spectrum Doppler*delay
        self.amplitude: amplitude of conjugate spectrum Doppler*delay
        self.phase: phase of conjugate spectrum Doppler*delay
        self.fD: Doppler rate
        self.tau: delay
        """
        pass
        
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_tau = self.CS.shape
        self.dfD = self.fD[1] - self.fD[0]
        self.dtau = self.tau[1] - self.tau[0]
        
    def mask(self,fD_min=-1.,fD_max=1.,tau_min=0.,tau_max=1.):
        assert tau_min>=0. and tau_max>0.
        bar = progressbar.ProgressBar(maxval=self.N_fD, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_fD in range(self.N_fD):
            bar.update(i_fD)
            fD = self.fD[i_fD]
            for i_tau in range(self.N_tau):
                tau = self.tau[i_tau]
                if tau >=0.:
                    if not (fD_min<fD<fD_max and tau_min<tau<tau_max):
                        self.CS[i_fD,i_tau] = 0.
                        #self.amplitude[i_fD,i_tau] = 0.
                        #self.phase[i_fD,i_tau] = 0.
                else:
                    if not (-fD_max<fD<-fD_min and -tau_max<tau<-tau_min):
                        self.CS[i_fD,i_tau] = 0.
        bar.finish()
        
        
        # upper = np.copy(self.CS)
        # upper[fD_min<self.fD,:] = 0.
        # upper[self.fD<fD_max,:] = 0.
        # upper[:,tau_min<self.tau] = 0.
        # upper[:,self.tau<tau_max] = 0.
        # lower = np.copy(self.CS)
        # lower[-fD_max<self.fD,:] = 0.
        # lower[self.fD<-fD_min,:] = 0.
        # lower[:,-tau_max<self.tau] = 0.
        # lower[:,self.tau<-tau_min] = 0.
        
        # self.CS = upper+lower
        self.phase = np.angle(self.CS)
        self.amplitude = np.abs(self.CS)
        
    def smartmask_stripe(self,fD0=0.,tau0=0.,fD_width_inner=1.,fD_width_outer=2.,tau_width_inner=1.,tau_width_outer=2.):
        assert tau0>=0.
        fD_freq = np.pi/2./(fD_width_outer-fD_width_inner)
        tau_freq = np.pi/2./(tau_width_outer-tau_width_inner)
        
        def mask(dfD,dtau):
            if dfD>fD_width_outer:
                factor = 0.
            elif dfD>fD_width_inner:
                factor = np.cos(fD_freq*(dfD-fD_width_inner))**2
            else:
                factor = 1.
            if dtau>tau_width_outer:
                factor *= 0.
            elif dtau>tau_width_inner:
                factor *= np.cos(tau_freq*(dtau-tau_width_inner))**2
            return factor
        
        arr = np.zeros((self.N_fD,self.N_tau))
        bar = progressbar.ProgressBar(maxval=self.N_fD, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_fD,fD in enumerate(self.fD):
            bar.update(i_fD)
            if np.abs(fD)<fD_width_outer:
                for i_tau,tau in enumerate(self.tau):
                    if np.abs(tau)<tau_width_outer:
                        arr[i_fD,i_tau] = mask(np.abs(fD),np.abs(tau))
        bar.finish()
        
        di_fD = int(np.rint(fD0/self.dfD))
        di_tau = int(np.rint(tau0/self.dtau))
        f_mask = np.roll(arr,(di_fD,di_tau),axis=(0,1)) + np.roll(arr,(-di_fD,-di_tau),axis=(0,1))
        f_mask[f_mask>1.] = 1.
        
        self.CS = self.CS*f_mask
        
        self.phase = np.angle(self.CS)
        self.amplitude = np.abs(self.CS)
        
    def smartmask_arclet(self,fD0=0.,tau0=0.,eta=1.,fD_width_inner=1.,fD_width_outer=2.,tau_width_inner=1.,tau_width_outer=2.):
        assert tau0>=0.
        fD_freq = np.pi/2./(fD_width_outer-fD_width_inner)
        tau_freq = np.pi/2./(tau_width_outer-tau_width_inner)
        
        def mask(dfD,dtau):
            if dfD>fD_width_outer:
                factor = 0.
            elif dfD>fD_width_inner:
                factor = np.cos(fD_freq*(dfD-fD_width_inner))**2
            else:
                factor = 1.
            if dtau>tau_width_outer:
                factor *= 0.
            elif dtau>tau_width_inner:
                factor *= np.cos(tau_freq*(dtau-tau_width_inner))**2
            return factor
        
        arr = np.zeros((self.N_fD,self.N_tau))
        bar = progressbar.ProgressBar(maxval=self.N_fD, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_fD,fD in enumerate(self.fD):
            bar.update(i_fD)
            if np.abs(fD)<fD_width_outer:
                for i_tau,tau in enumerate(self.tau):
                    if np.abs(tau)<tau_width_outer:
                        arr[i_fD,i_tau] = mask(np.abs(fD),np.abs(tau))
        bar.finish()
        for i_fD,fD in enumerate(self.fD):
            col = arr[i_fD,:]
            shift = int(np.rint(eta*fD**2/self.dtau))
            col = np.roll(col,shift)
            arr[i_fD,:] = col
        
        di_fD = int(np.rint(fD0/self.dfD))
        di_tau = int(np.rint(tau0/self.dtau))
        f_mask = np.roll(np.flip(arr),(di_fD,di_tau),axis=(0,1)) + np.roll(arr,(-di_fD,-di_tau),axis=(0,1))
        f_mask[f_mask>1.] = 1.
        
        self.CS = self.CS*f_mask
        
        self.phase = np.angle(self.CS)
        self.amplitude = np.abs(self.CS)
        
class pulse_profile:
    type = "pulse profile"
    
    def __init__(self):
        """
        override this
        self.pulse: pulse profile time*frequency
        self.t: pulse phase in seconds
        self.nu: frequency
        """
        pass
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.pulse.shape
        self.dt = self.t[1]-self.t[0]
        self.dnu = self.nu[1]-self.nu[0]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]
        self.nu_min = self.nu[0]
        self.nu_max = self.nu[-1]
        self.timespan = self.t_max-self.t_min
        self.bandwidth = self.nu_max-self.nu_min
        
class generic_intensity(intensity):
    def __init__(self,t,nu,DS,mjd0):
        self.t = t
        self.nu = nu
        self.DS = DS
        self.mjd = mjd0 + (self.t-np.mean(self.t))/day
        self.recalculate()
        
class generic_masked_intensity(dynspec_masked):
    def __init__(self,t,nu,DS,mask,mjd):
        self.t = t
        self.nu = nu
        self.DS = DS
        self.mask = mask
        self.mjd = mjd
        masked_DS = np.copy(self.DS)
        masked_DS[self.mask] = np.nan
        self.DS = self.DS/np.nanstd(masked_DS)
        self.recalculate()
        
class generic_visibility(visibility):
    def __init__(self,t,nu,DS,phase,mjd0):
        self.t = t
        self.nu = nu
        self.DS = DS
        self.phase = phase
        self.mjd = mjd0 + (self.t-np.mean(self.t))/day
        self.recalculate()

class SecSpec_FFT(secondary_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"SecSpec_FFT.npz")
            if DS!=None: #not os.path.exists(file_data):
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,SS=self.SS)
            else:
                lib_data = np.load(file_data)
                self.fD = lib_data["fD"]
                self.tau = lib_data["tau"]
                self.SS = lib_data["SS"]
        self.recalculate()
        
    def compute(self,DS,kwargs):
        if (DS.type == "intensity") or (DS.type == "secondary dynamic spectrum"):
            #- prepare data
            data = DS.DS - np.mean(DS.DS)
            self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
            self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
            conj_spec = np.fft.fftshift(np.fft.fft2(data,axes=(0,1)),axes=(0,1))
            self.SS = np.abs(conj_spec)**2
        elif DS.type == "visibility":
            #- prepare data
            data = DS.DS*np.exp(1j*DS.phase)
            self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
            self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
            conj_spec = np.fft.fftshift(np.fft.fft2(data,axes=(0,1)),axes=(0,1))
            self.SS = np.abs(conj_spec)**2
        else:
            raise TypeError
            
class SecSpec_zeta(secondary_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"SecSpec_zeta.npz")
            if DS!=None: #not os.path.exists(file_data):
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,SS=self.SS)
            else:
                lib_data = np.load(file_data)
                self.fD = lib_data["fD"]
                self.tau = lib_data["tau"]
                self.SS = lib_data["SS"]
        self.recalculate()
        
    def compute(self,DS,kwargs):
        self.zeta = kwargs.get("zeta",1.)
        t0 = kwargs.get("t0",DS.t0)
        if (DS.type == "intensity") or (DS.type == "secondary dynamic spectrum"):
            #- prepare data
            data = (DS.DS - np.mean(DS.DS))*np.exp(-2.0j*np.pi*self.zeta**2*DS.nu[na,:]*(DS.t[:,na]-t0)**2)
            self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
            self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
            conj_spec = np.fft.fftshift(np.fft.fft2(data,axes=(0,1)),axes=(0,1))
            self.SS = np.abs(conj_spec)**2
        elif DS.type == "visibility":
            #- prepare data
            data = DS.DS*np.exp(1j*DS.phase)*np.exp(-2.0j*np.pi*self.zeta**2*DS.nu[na,:]*(DS.t[:,na]-t0)**2)
            self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
            self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
            conj_spec = np.fft.fftshift(np.fft.fft2(data,axes=(0,1)),axes=(0,1))
            self.SS = np.abs(conj_spec)**2
        else:
            raise TypeError

class SecSpec_NuT(secondary_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SecSpec_NuT.npz")
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,SS=self.SS,nu0=self.nu0,mjd0=self.mjd0)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.tau = lib_data["tau"]
                    self.SS = lib_data["SS"]
                    self.nu0 = lib_data.get("nu0",None)
                    self.mjd0 = lib_data.get("mjd0",None)
                else:
                    raise KeyError
        self.recalculate()
    
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        subtract_mean = kwargs.get("subtract_mean",True)
        self.mjd0 = kwargs.get("mjd0",DS.mjd0)
        self.nu0 = kwargs.get("nu0",DS.nu0)
        self.f_refr = kwargs.get("f_refr",0.)
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        #- prepare data
        data = DS.DS
        if subtract_mean:
            data = data - np.mean(DS.DS)
        #tt = (DS.t-DS.t0)/self.nu0
        hss_real = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        hss_im = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        if self.f_refr==0.:
            tt = (DS.t-DS.t0+(DS.mjd0-self.mjd0)*day)/self.nu0
            lib.NuT(DS.N_t,DS.N_nu,DS.N_t,tt.astype('float64'),DS.nu.astype('float64'),self.fD.astype('float64'),data.astype('float64').flatten(),hss_real,hss_im)
        else:
            tt = DS.t-DS.t0+(DS.mjd0-self.mjd0)*day-self.f_refr/self.nu0**2
            lib.NuT_derefracted(DS.N_t,DS.N_nu,DS.N_t,self.nu0,self.f_refr,tt.astype('float64'),DS.nu.astype('float64'),self.fD.astype('float64'),data.astype('float64').flatten(),hss_real,hss_im)
        hss = hss_real.reshape((DS.N_t,DS.N_nu))+1.j*hss_im.reshape((DS.N_t,DS.N_nu))
        self.SS = np.abs(np.fft.fftshift(np.fft.fft(hss,axis=1),axes=1))**2
        
class SecSpec_grad(secondary_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SecSpec_grad.npz")
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,SS=self.SS,nu0=self.nu0,mjd0=self.mjd0)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.tau = lib_data["tau"]
                    self.SS = lib_data["SS"]
                    self.nu0 = lib_data.get("nu0",None)
                    self.mjd0 = lib_data.get("mjd0",None)
                else:
                    raise KeyError
        self.recalculate()
    
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        subtract_mean = kwargs.get("subtract_mean",True)
        self.mjd0 = kwargs.get("mjd0",DS.mjd0)
        self.nu0 = kwargs.get("nu0",DS.nu0)
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        #- prepare data
        data = DS.DS
        if subtract_mean:
            data = data - np.mean(DS.DS)
        #tt = (DS.t-DS.t0)/self.nu0
        ss_real = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        ss_im = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        tt = (DS.t-DS.t0+(DS.mjd0-self.mjd0)*day)
        #nu1 = -2.*np.pi*self.nu0**2/DS.nu
        #nu3 = -2.*np.pi*self.nu0**4/DS.nu**3
        lib.SSgrad(DS.N_t,DS.N_nu,tt.astype('float64'),DS.nu.astype('float64'),self.fD.astype('float64'),self.tau.astype('float64'),self.nu0,data.astype('float64').flatten(),ss_real,ss_im)
        self.SS = np.abs(ss_real.reshape((DS.N_t,DS.N_nu))+1.j*ss_im.reshape((DS.N_t,DS.N_nu)))**2
        
class SecSpec_refr():
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_fnum2 = self.SS.shape
        self.dfD = self.fD[1] - self.fD[0]
        self.dfnum2 = self.fnum2[1] - self.fnum2[0]
        self.fD_max = np.max(self.fD)
        self.fnum2_max = np.max(self.fnum2)
        
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SecSpec_refr.npz")
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,fnum2=self.fnum2,SS=self.SS)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.fnum2 = lib_data["fnum2"]
                    self.SS = lib_data["SS"]
                else:
                    raise KeyError
        self.recalculate()
    
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        subtract_mean = kwargs.get("subtract_mean",True)
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        num2 = 1./DS.nu**2
        dnum2 = np.diff(num2).mean()
        self.fnum2 = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,dnum2))
        #- prepare data
        data = DS.DS
        if subtract_mean:
            data = data - np.mean(DS.DS)
        #tt = (DS.t-DS.t0)/self.nu0
        hss_real = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        hss_im = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        lib.SS_refr(DS.N_t,DS.N_nu,DS.N_nu,DS.t.astype('float64'),num2.astype('float64'),self.fnum2.astype('float64'),data.astype('float64').flatten(),hss_real,hss_im)
        hss = hss_real.reshape((DS.N_t,DS.N_nu))+1.j*hss_im.reshape((DS.N_t,DS.N_nu))
        self.SS = np.abs(np.fft.fftshift(np.fft.fft(hss,axis=0),axes=0))**2
        
class SecSpec_Lambda(secondary_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SecSpec_Lambda.npz")
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,Beta=self.Beta,SS=self.SS)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.Beta = lib_data["Beta"]
                    self.fD = lib_data["fD"]
                    self.tau = lib_data["tau"]
                    self.SS = lib_data["SS"]
                else:
                    raise KeyError
        self.recalculate()
    
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        #self.Lambda = kwargs.get("Lambda",v_c/DS.nu0)
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        #- prepare data
        data = DS.DS - np.mean(DS.DS)
        L = v_c/DS.nu
        nu_L = np.linspace(L[-1],L[0],num=DS.N_nu,endpoint=True)
        #L_tau = self.tau * (v_c/self.Lambda**2)
        self.Beta = np.fft.fftshift(np.fft.fftfreq(len(nu_L),nu_L[1]-nu_L[0]))
        
        for i_t in range(DS.N_t):
            ip = interp.interp1d(L,data[i_t,:])
            data[i_t,:] = ip(nu_L)
            data[i_t,:] = data[i_t,::-1]
        self.SS = np.abs(np.fft.fftshift(np.fft.fft2(data,axes=(0,1)),axes=(0,1)))**2
        
        # hss_real = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        # hss_im = np.zeros((DS.N_t*DS.N_nu),dtype='float64')
        # lib.Lambda(DS.N_t,DS.N_nu,DS.N_nu,DS.t.astype('float64'),L.astype('float64'),L_tau.astype('float64'),data.astype('float64').flatten(),hss_real,hss_im)
        # hss = hss_real.reshape((DS.N_t,DS.N_nu))+1.j*hss_im.reshape((DS.N_t,DS.N_nu))
        # self.SS = np.abs(np.fft.fftshift(np.fft.fft(hss,axis=0),axes=0))**2
        
class DynSpec_patchnorm(intensity):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","DynSpec_patchnorm.npz")
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,nu=self.nu,t=self.t,mjd=self.mjd,DS=self.DS)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.nu = lib_data["nu"]
                    self.t = lib_data["t"]
                    self.mjd = lib_data["mjd"]
                    self.DS = lib_data["DS"]
                else:
                    raise KeyError
        self.recalculate()
    
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        fD_min = kwargs.get("fD_min",0.*mHz)
        fD_max = kwargs.get("fD_max",1.*mHz)
        tau_min = kwargs.get("tau_min",0.*mus)
        tau_max = kwargs.get("tau_max",1.*mus)
        
        def find_chunks(N,Nc):
            """
            N : length of list
            Nc : Maximum number of entries per chunk
            """
            #number of chunks
            Ns = (2*N)//Nc
            #optimal shift of chunk (float)
            shift = N/(Ns+1.)
            #create list by rounding to integers
            starts = [np.rint(i*shift).astype(int) for i in range(Ns)]
            #mids = [np.rint((i+1)*shift).astype(int) for i in range(Ns)]
            ends = [np.rint((i+2)*shift).astype(int) for i in range(Ns)]
            return starts,ends,int(shift)
        
        self.DS = np.zeros_like(DS.DS)
        #- split into overlapping chunks
        t_starts,t_ends,t_shift = find_chunks(DS.N_t,tchunk)
        nu_starts,nu_ends,nu_shift = find_chunks(DS.N_nu,nuchunk)
        N_tchunk = len(t_starts)
        N_nuchunk = len(nu_starts)
            
        #main computation
        bar = progressbar.ProgressBar(maxval=N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for fc in range(N_nuchunk):
            bar.update(fc)
            #select Chunk
            l_nuchunk = nu_ends[fc]-nu_starts[fc]
            nu = DS.nu[nu_starts[fc]:nu_ends[fc]]
            # combine by windowing
            fmsk = np.ones(l_nuchunk)
            if fc>0:
                fmsk[:nu_shift] = np.sin((np.pi/2)*np.linspace(0,nu_shift-1,nu_shift)/nu_shift)**2
            if fc<N_nuchunk-1:
                fmsk[-nu_shift:] = np.cos((np.pi/2)*np.linspace(0,nu_shift-1,nu_shift)/nu_shift)**2
            for tc in range(N_tchunk):
                t = DS.t[t_starts[tc]:t_ends[tc]]
                l_tchunk = t_ends[tc]-t_starts[tc]
                tmsk = np.ones(l_tchunk)
                dspec = DS.DS[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]]
                #dspec = dspec - np.mean(dspec)
                #compute secondary spectrum
                SS = np.fft.fftshift(np.fft.fft2(dspec))
                fD = np.fft.fftshift(np.fft.fftfreq(len(t),DS.dt))
                tau = np.fft.fftshift(np.fft.fftfreq(len(nu),DS.dnu))
                #normalize by mean amplitude of chosen region in SecSpec
                is_fD = np.squeeze(np.argwhere(np.logical_and(fD<fD_max,fD>fD_min)))
                is_tau = np.squeeze(np.argwhere(np.logical_and(tau<tau_max,tau>tau_min)))
                #print(is_fD)
                #print(is_tau)
                SS_region = np.abs(SS)[is_fD,:]
                SS_region = SS_region[:,is_tau]
                norm = np.mean(SS_region)
                if norm>0.:
                    model = dspec/norm
                else:
                    model = dspec*0.
                #Combine chunks
                if tc>0:
                    tmsk[:t_shift] = np.sin((np.pi/2)*np.linspace(0,t_shift-1,t_shift)/t_shift)**2
                if tc<N_tchunk-1:
                    tmsk[-t_shift:] = np.cos((np.pi/2)*np.linspace(0,t_shift-1,t_shift)/t_shift)**2
                self.DS[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]] += model*tmsk[:,na]*fmsk[na,:]
        bar.finish()
        
        self.t = DS.t
        self.nu = DS.nu
        self.mjd = DS.mjd
        self.DS = self.DS/np.std(self.DS)
        
class staufD:
    type = "staufD diagram"
    
    def __init__(self,SS,**kwargs):
        """
        self.staufD: staufD diagram Doppler*sqrt(delay)
        self.fD: Doppler rate
        self.stau: sqrt(delay)
        """
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","staufD.npz")
        if self.data_path==None:
           self.compute(SS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if SS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(SS,kwargs)
                np.savez(file_data,fD=self.fD,stau=self.stau,staufD=self.staufD)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.stau = lib_data["stau"]
                    self.staufD = lib_data["staufD"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_stau = self.staufD.shape
        self.stau_max = np.max(self.stau)
        self.stau_min = np.min(self.stau)
        self.fD_max = np.max(self.fD)
        self.fD_min = np.min(self.fD)
        self.dfD = np.diff(self.fD).mean()
        self.dstau = np.diff(self.stau).mean()
        
    def compute(self,SS,kwargs):
        #load and check data
        if not SS.type == "secondary spectrum":
            raise TypeError
        self.N_stau = kwargs.get("N_stau",100)
        tau_max = kwargs.get("tau_max",np.max(SS.tau))
        rnoise = kwargs.get("remove_noise",True)
        data = np.sqrt(SS.SS)
        self.fD = SS.fD
        
        #preparations
        noise = np.median(data)
        stau_max = np.sqrt(tau_max)
        self.stau = np.linspace(-stau_max,stau_max,num=self.N_stau,endpoint=True)
        dstau = self.stau[1]-self.stau[0]
        stau_u = self.stau+dstau/2.
        stau_l = self.stau-dstau/2.
        taus = self.stau**2*np.sign(self.stau)
        taus_1 = stau_u**2*np.sign(stau_u)
        taus_2 = stau_l**2*np.sign(stau_l)
        taus_u = np.maximum(taus_1,taus_2)
        taus_l = np.minimum(taus_1,taus_2)
        # - get pixel boundaries of SS
        ltau = SS.tau - SS.dtau/2.
        rtau = SS.tau + SS.dtau/2.
        
        #create containers
        self.staufD = np.zeros((SS.N_fD,self.N_stau),dtype=float)
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=self.N_stau, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_stau in range(self.N_stau):
            bar.update(i_stau)
            # - determine indices of boundary pixels
            i_tau_l = np.argmax(rtau>taus_l[i_stau])
            i_tau_u = np.argmax(rtau>taus_u[i_stau])
            # - sum pixels
            for i_tau in range(i_tau_l,i_tau_u+1):
                length = (np.min([taus_u[i_stau],rtau[i_tau]])-np.max([taus_l[i_stau],ltau[i_tau]]))/SS.dtau
                self.staufD[:,i_stau] += data[:,i_tau]*length
                if rnoise:
                    self.staufD[:,i_stau] -= noise*length
        bar.finish()
        
    def get_eta(self,**kwargs):
        vmin = kwargs.get("vmin",None)
        vmax = kwargs.get("vmax",None)
        xmin = kwargs.get("xmin",self.stau_min/1.0e-3)
        xmax = kwargs.get("xmax",self.stau_max/1.0e-3)
        ymin = kwargs.get("ymin",self.fD_min/1.0e-3)
        ymax = kwargs.get("ymax",self.fD_max/1.0e-3)
        nu0_data = kwargs.get("nu0_data",1.4e+9)
        eta_max = kwargs.get("eta_max",0.35)
        eta_init= kwargs.get("eta_init",0.03)
    
        def plot_staufD(ax,fd,stau,staufD,nx,ny):
            sampling_fd = np.max([int(len(fd)/ny),1])
            sampling_stau = np.max([int(len(stau)/nx),1])
            data_staufD = block_reduce(staufD, block_size=(sampling_fd,sampling_stau), func=np.mean)
            coordinates = np.array([fd,fd])
            coordinates = block_reduce(coordinates, block_size=(1,sampling_fd), func=np.mean, cval=fd[-1])
            data_fd = coordinates[0,:]
            coordinates = np.array([stau,stau])
            coordinates = block_reduce(coordinates, block_size=(1,sampling_stau), func=np.mean, cval=stau[-1])
            data_stau = coordinates[0,:]
            min_nonzero = np.min(data_staufD[np.nonzero(data_staufD)])
            data_staufD[data_staufD == 0] = min_nonzero
            data_staufD = np.log10(data_staufD)
            im = ax.pcolormesh(data_stau,data_fd,data_staufD,cmap='viridis',vmin=vmin,vmax=vmax)
            plt.colorbar(im,ax=ax)
    
        #set up the canvas
        plot_width = 1200
        plot_height = 900
        plot_dpi = 100
        plot_bottom = 0.15
        plot_top = 0.93
        plot_left = 0.10
        plot_right = 0.95
        plot_wspace = 0.2
        plot_hspace = 0.2
        figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
        plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
        ax = figure.add_subplot(1,1,1)
        plot_staufD(ax,self.fD/1.0e-3,self.stau/1.0e-3,self.staufD,500,500)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_xlabel(r"$\sqrt{\tau}$ [$\sqrt{\mu s}$]")
        ax.set_ylabel(r"$f_{\rm D}$ [mHz]")
        slider_eta = mpl.widgets.Slider(plt.axes([0.12,0.02,0.7,0.03]),r'$\eta$',0.0,0.35,valinit=0.03)
        slider_err = mpl.widgets.Slider(plt.axes([0.53,0.07,0.4,0.03]),r'$\sigma_{\eta}$ %',0.0,50.0,valinit=0.0)
        y_fit = np.linspace(ymin,ymax,num=201,endpoint=True)
        corr_factor = 1.4e+9/nu0_data
        x_u = np.abs(y_fit)*np.sqrt(slider_eta.val*(1.+slider_err.val/100.))*corr_factor
        x_d = np.abs(y_fit)*np.sqrt(slider_eta.val*(1.-slider_err.val/100.))*corr_factor
        fit_plot_u, = ax.plot(x_u,y_fit,color='red',linestyle='-',markersize=0,alpha=0.5)
        fit_plot_d, = ax.plot(x_d,y_fit,color='red',linestyle='-',markersize=0,alpha=0.5)
        
        def update_eta(event):
            x_u = np.abs(y_fit)*np.sqrt(slider_eta.val*(1.+slider_err.val/100.))*corr_factor
            x_d = np.abs(y_fit)*np.sqrt(slider_eta.val*(1.-slider_err.val/100.))*corr_factor
            fit_plot_u.set_xdata(x_u)
            fit_plot_d.set_xdata(x_d)
            figure.canvas.draw_idle()
        
        slider_eta.on_changed(update_eta)
        slider_err.on_changed(update_eta)
        plt.show()
        plt.clf()
        
        return slider_eta.val,slider_eta.val*slider_err.val/100.
        
    def get_zeta(self,**kwargs):
        vmin = kwargs.get("vmin",None)
        vmax = kwargs.get("vmax",None)
        xmin = kwargs.get("xmin",self.fD_min/1.0e-3)
        xmax = kwargs.get("xmax",self.fD_max/1.0e-3)
        ymin = kwargs.get("ymin",self.stau_min/1.0e-3)
        ymax = kwargs.get("ymax",self.stau_max/1.0e-3)
        nu0_data = kwargs.get("nu0_data",1.4e+9)
        zeta_max = kwargs.get("zeta_max",0.35)
        zeta_init= kwargs.get("zeta_init",0.03)
    
        def plot_staufD(ax,fd,stau,staufD,nx,ny):
            sampling_fd = np.max([int(len(fd)/ny),1])
            sampling_stau = np.max([int(len(stau)/nx),1])
            data_staufD = block_reduce(staufD, block_size=(sampling_fd,sampling_stau), func=np.mean)
            coordinates = np.array([fd,fd])
            coordinates = block_reduce(coordinates, block_size=(1,sampling_fd), func=np.mean, cval=fd[-1])
            data_fd = coordinates[0,:]
            coordinates = np.array([stau,stau])
            coordinates = block_reduce(coordinates, block_size=(1,sampling_stau), func=np.mean, cval=stau[-1])
            data_stau = coordinates[0,:]
            #min_nonzero = np.min(data_staufD[np.nonzero(data_staufD)])
            #min_nonzero = np.min(data_staufD[data_staufD>0.])
            #print(min_nonzero)
            #data_staufD[data_staufD <= 0.] = min_nonzero
            if vmin!=None:
                minvalue = np.power(10.,vmin)
            else:
                minvalue = np.min(data_staufD[data_staufD>0.])
            data_staufD[data_staufD < minvalue] = minvalue
            data_staufD = np.log10(data_staufD)
            data_staufD = np.swapaxes(data_staufD,0,1)
            im = ax.pcolormesh(data_fd,data_stau,data_staufD,cmap='viridis',vmin=vmin,vmax=vmax,shading='nearest')
            plt.colorbar(im,ax=ax)
    
        #set up the canvas
        plot_width = 1200
        plot_height = 900
        plot_dpi = 100
        plot_bottom = 0.15
        plot_top = 0.99
        plot_left = 0.10
        plot_right = 0.99
        plot_wspace = 0.2
        plot_hspace = 0.2
        textsize=22
        labelsize=24
        figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
        plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
        mpl.rcParams.update({"axes.labelsize": labelsize})
        mpl.rcParams.update({"axes.titlesize": labelsize})
        mpl.rcParams.update({"xtick.labelsize": textsize})
        mpl.rcParams.update({"ytick.labelsize": textsize})
        mpl.rcParams.update({"font.size": labelsize})
        mpl.rcParams.update({"legend.fontsize": textsize})
        ax = figure.add_subplot(1,1,1)
        plot_staufD(ax,self.fD/1.0e-3,self.stau/1.0e-3,self.staufD,500,500)
        ax.set_xlim([xmin,xmax])
        ax.set_ylim([ymin,ymax])
        ax.set_ylabel(r"$\sqrt{\tau}$ [$\sqrt{\mu s}$]")
        ax.set_xlabel(r"$f_{\rm D}$ [mHz]")
        #slider_zeta = mpl.widgets.Slider(plt.axes([0.12,0.02,0.7,0.03]),r'$\zeta$',0.0,zeta_max,valinit=zeta_init)
        #slider_err = mpl.widgets.Slider(plt.axes([0.53,0.07,0.4,0.03]),r'$\sigma_{\zeta}$ %',0.0,50.0,valinit=0.0)
        slider_zeta = mpl.widgets.Slider(plt.axes([0.15,0.02,0.65,0.03]),r'$\partial_t \sqrt{\tau}$ [s$^{-1/2}$]',0.0,zeta_max,valinit=zeta_init)
        slider_err = mpl.widgets.Slider(plt.axes([0.62,0.07,0.25,0.03]),r'$\sigma$ [%]',0.0,50.0,valinit=0.0)
        slider_zeta.label.set_fontsize(20)
        slider_err.label.set_fontsize(20)
        y_fit = np.linspace(ymin,ymax,num=201,endpoint=True)
        x_u = np.abs(y_fit)*2.*nu0_data*(slider_zeta.val*(1.+slider_err.val/100.))
        x_d = np.abs(y_fit)*2.*nu0_data*(slider_zeta.val*(1.-slider_err.val/100.))
        fit_plot_u, = ax.plot(x_u,y_fit,color='white',linestyle='-',markersize=0,alpha=0.5,linewidth=3)
        fit_plot_d, = ax.plot(x_d,y_fit,color='white',linestyle='-',markersize=0,alpha=0.5,linewidth=3)
        
        def update_zeta(event):
            x_u = np.abs(y_fit)*2.*nu0_data*(slider_zeta.val*(1.+slider_err.val/100.))
            x_d = np.abs(y_fit)*2.*nu0_data*(slider_zeta.val*(1.-slider_err.val/100.))
            fit_plot_u.set_xdata(x_u)
            fit_plot_d.set_xdata(x_d)
            figure.canvas.draw_idle()
        
        slider_zeta.on_changed(update_zeta)
        slider_err.on_changed(update_zeta)
        plt.show()
        plt.clf()
        
        return slider_zeta.val,slider_zeta.val*slider_err.val/100.
        
        
class thfD:
    type = "thfD diagram"
    
    def __init__(self,staufD,**kwargs):
        """
        self.thfD: staufD diagram theta*doppler
        self.fD: Doppler rate
        self.th: angular position
        """
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
           self.compute(staufD,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"thfD.npz")
            if staufD!=None:
                self.compute(staufD,kwargs)
                np.savez(file_data,fD=self.fD,theta=self.th,thfD=self.thfD)
            else:
                lib_data = np.load(file_data)
                self.fD = lib_data["fD"]
                self.th = lib_data["theta"]
                self.thfD = lib_data["thfD"]
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_th,self.N_fD = self.thfD.shape
        
    def compute(self,staufD,kwargs):
        #load and check data
        if not staufD.type == "staufD diagram":
            raise TypeError
        self.Deff = kwargs.get("Deff",120.*pc)
        self.veff = kwargs.get("veff",20.*kms)
        self.nu0 = kwargs.get("nu0",1400.*MHz)
        width_fD = kwargs.get("width_fD",1.*mHz)
            
        # - compute thfD coordinates
        self.th = -staufD.stau/np.sqrt(self.Deff/(2.*v_c))
        fD_c = -self.nu0/v_c*np.abs(self.veff*self.th)
        # - determine indices of f_D range
        dfd = np.diff(staufD.fD).mean()
        N_fd = int(np.ceil(width_fD/dfd/2.))
        fD_u = fD_c + N_fd*dfd
        fD_l = fD_c - N_fd*dfd
        lfd = staufD.fD - dfd/2.
        rfd = staufD.fD + dfd/2.
        self.fD = np.linspace(-N_fd*dfd,N_fd*dfd,num=1+2*N_fd,endpoint=True)
        # - create thfD diagram
        N_th = len(self.th)
        self.thfD = np.zeros((N_th,1+2*N_fd),dtype=float)
        for i_th in range(N_th):
            i_fD_l = np.argmax(rfd>fD_l[i_th])
            i_fD_u = np.argmax(rfd>fD_u[i_th])
            if staufD.stau[i_th]>0.:
                self.thfD[i_th,:] = staufD.staufD[i_fD_l:i_fD_u+1,i_th]
            else:
                self.thfD[i_th,:] = staufD.staufD[i_fD_u:i_fD_l-1:-1,i_th]
                
class staufD_arc:
    type = "thfD diagram on arc"
    
    def __init__(self,SS,**kwargs):
        """
        self.staufD: staufD diagram Doppler*sqrt(delay)
        self.fD: Doppler rate
        self.stau: sqrt(delay)
        """
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","staufD_arc.npz")
        if self.data_path==None:
           self.compute(SS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if SS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(SS,kwargs)
                np.savez(file_data,fD=self.fD,stau=self.stau,staufD=self.staufD)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.stau = lib_data["stau"]
                    self.staufD = lib_data["staufD"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_stau = self.staufD.shape
        self.stau_max = np.max(self.stau)
        self.stau_min = np.min(self.stau)
        self.fD_max = np.max(self.fD)
        self.fD_min = np.min(self.fD)
        self.dfD = np.diff(self.fD).mean()
        self.dstau = np.diff(self.stau).mean()
        
    def compute(self,SS,kwargs):
        #load and check data
        if SS.type == "secondary spectrum":
            data = np.sqrt(SS.SS)
        elif SS.type == "wavefield":
            data = SS.amplitude
            # for i_fD in range(SS.N_fD):
                # if SS.fD[i_fD] < 0:
                    # data[i_fD,:] = np.flip(data[i_fD,:])
        else:
            raise TypeError
        self.N_stau = kwargs.get("N_stau",100)
        self.N_fD = kwargs.get("N_fD",100)
        self.eta = kwargs.get("eta",0.)
        self.zeta = kwargs.get("zeta",0.)
        tau_max = kwargs.get("tau_max",np.max(SS.tau))
        width_fD = kwargs.get("width_fD",1.*mHz)
        rnoise = kwargs.get("remove_noise",True)
        nu0 = kwargs.get("nu0",1.4e+9)
        self.fD = SS.fD
        
        #preparations
        noise = np.median(data)
        # - compute tau coordinates
        stau_max = np.sqrt(tau_max)
        self.stau = np.linspace(-stau_max,stau_max,num=self.N_stau,endpoint=True)
        dstau = self.stau[1]-self.stau[0]
        stau_u = self.stau+dstau/2.
        stau_l = self.stau-dstau/2.
        # taus = self.stau**2*np.sign(self.stau)
        # taus_1 = stau_u**2*np.sign(stau_u)
        # taus_2 = stau_l**2*np.sign(stau_l)
        taus = self.stau**2
        taus_1 = stau_u**2
        taus_2 = stau_l**2
        taus_u = np.maximum(taus_1,taus_2)
        taus_l = np.minimum(taus_1,taus_2)
        ltau = SS.tau - SS.dtau/2.
        rtau = SS.tau + SS.dtau/2.
        # - compute fD coordinates
        lfD = SS.fD - SS.dfD/2.
        rfD = SS.fD + SS.dfD/2.
        self.fD = np.linspace(-width_fD/2.,width_fD/2.,num=self.N_fD,endpoint=True)
        dfD = np.diff(self.fD).mean()
        fD_u = self.fD+dfD/2.
        fD_l = self.fD-dfD/2.
        
        #create containers
        self.staufD = np.zeros((self.N_fD,self.N_stau),dtype=float)
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=self.N_stau, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        # for i_stau in range(self.N_stau):
            # bar.update(i_stau)
            # # - determine indices of boundary pixels
            # i_tau_l = np.argmax(rtau>taus_l[i_stau])
            # i_tau_u = np.argmax(rtau>taus_u[i_stau])
            # for i_fD_arc in range(self.N_fD):
                # rfD_arc = rfD - np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                # lfD_arc = lfD - np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                # i_fD_l = np.argmax(rfD_arc>fD_l[i_fD_arc])
                # i_fD_u = np.argmax(rfD_arc>fD_u[i_fD_arc])
                # # - sum pixels
                # for i_tau in range(i_tau_l,i_tau_u+1):
                    # for i_fD in range(i_fD_l,i_fD_u+1):
                        # l_tau = (np.min([taus_u[i_stau],rtau[i_tau]])-np.max([taus_l[i_stau],ltau[i_tau]]))/SS.dtau
                        # l_fD = (np.min([fD_u[i_fD_arc],rfD_arc[i_fD]])-np.max([fD_l[i_fD_arc],lfD_arc[i_fD]]))/SS.dfD
                        # self.staufD[i_fD_arc,i_stau] += data[i_fD,i_tau]*l_tau*l_fD
                        # if rnoise:
                            # self.staufD[i_fD_arc,i_stau] -= noise*l_tau*l_fD
        for i_stau in range(self.N_stau):
            bar.update(i_stau)
            for i_fD_arc in range(self.N_fD):
                # - determine indices of boundary pixels
                if self.stau[i_stau] > 0.:
                    i_tau_l = np.argmax(rtau>taus_l[i_stau])
                    i_tau_u = np.argmax(rtau>taus_u[i_stau])
                    if not self.eta==0.:
                        rfD_arc = rfD - np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                        lfD_arc = lfD - np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                    else:
                        rfD_arc = rfD - np.abs(self.stau[i_stau]*(2.*nu0*self.zeta))
                        lfD_arc = lfD - np.abs(self.stau[i_stau]*(2.*nu0*self.zeta))
                    i_fD_l = np.argmax(rfD_arc>fD_l[i_fD_arc])
                    i_fD_u = np.argmax(rfD_arc>fD_u[i_fD_arc])
                else:
                    i_tau_l = np.argmax(rtau>taus_l[i_stau])
                    i_tau_u = np.argmax(rtau>taus_u[i_stau])
                    if not self.eta==0.:
                        rfD_arc = rfD + np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                        lfD_arc = lfD + np.abs(self.stau[i_stau]/np.sqrt(self.eta))
                    else:
                        rfD_arc = rfD + np.abs(self.stau[i_stau]*(2.*nu0*self.zeta))
                        lfD_arc = lfD + np.abs(self.stau[i_stau]*(2.*nu0*self.zeta))
                    i_fD_l = np.argmax(rfD_arc>fD_l[i_fD_arc])
                    i_fD_u = np.argmax(rfD_arc>fD_u[i_fD_arc])
                # - sum pixels
                for i_tau in range(i_tau_l,i_tau_u+1):
                    for i_fD in range(i_fD_l,i_fD_u+1):
                        l_tau = (np.min([taus_u[i_stau],rtau[i_tau]])-np.max([taus_l[i_stau],ltau[i_tau]]))/SS.dtau
                        l_fD = (np.min([fD_u[i_fD_arc],rfD_arc[i_fD]])-np.max([fD_l[i_fD_arc],lfD_arc[i_fD]]))/SS.dfD
                        self.staufD[i_fD_arc,i_stau] += data[i_fD,i_tau]*l_tau*l_fD
                        if rnoise:
                            self.staufD[i_fD_arc,i_stau] -= noise*l_tau*l_fD
        bar.finish()
        
class thth_arc:
    type = "thth diagram on arc"
    
    def __init__(self,SS,**kwargs):
        """
        self.thth: thth diagram sqrt(delay)*sqrt(delay)
        self.thx: short axis in units of sqrt(delay)
        self.thy: long axis in units of sqrt(delay)
        """
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","thth_arc.npz")
        if self.data_path==None:
           self.compute(SS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if SS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(SS,kwargs)
                np.savez(file_data,thx=self.thx,thy=self.thy,thth=self.thth)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.thx = lib_data["thx"]
                    self.thy = lib_data["thy"]
                    self.thth = lib_data["thth"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_thx,self.N_thy = self.thth.shape
        self.thy_max = np.max(self.thy)
        self.thy_min = np.min(self.thy)
        self.thx_max = np.max(self.thx)
        self.thx_min = np.min(self.thx)
        self.dthx = np.diff(self.thx).mean()
        self.dthy = np.diff(self.thy).mean()
        
    def compute(self,SS,kwargs):
        #load and check data
        if SS.type == "secondary spectrum":
            data = np.sqrt(SS.SS)
        elif SS.type == "wavefield":
            data = SS.amplitude
            # for i_fD in range(SS.N_fD):
                # if SS.fD[i_fD] < 0:
                    # data[i_fD,:] = np.flip(data[i_fD,:])
        else:
            raise TypeError
        self.N_thy = kwargs.get("N_thy",100)
        self.N_thx = kwargs.get("N_thx",100)
        self.eta = kwargs.get("eta",0.)
        self.zeta = kwargs.get("zeta",0.)
        tau_max = kwargs.get("tau_max",np.max(SS.tau))
        width_fD = kwargs.get("width_fD",1.*mHz)
        rnoise = kwargs.get("remove_noise",True)
        nu0 = kwargs.get("nu0",1.4e+9)
        #self.fD = SS.fD
        
        #preparations
        if self.eta==0.:
            fD_to_stau = 1./(2.*nu0*self.zeta)
        elif self.zeta==0.:
            fD_to_stau = np.sqrt(self.eta)
        else:
            print("You cannot specify both eta and zeta!")
            raise KeyError
        # - compute tau coordinates
        stau_max = np.sqrt(tau_max)
        self.thy = np.linspace(-stau_max,stau_max,num=self.N_thy,endpoint=True)
        self.thx = np.linspace(-0.5*width_fD*fD_to_stau,0.5*width_fD*fD_to_stau,num=self.N_thx,endpoint=True)
        
        #preparations        
        # - create container
        self.thth = np.zeros((self.N_thx,self.N_thy),dtype=float)
        # - estimate noise
        noise = np.median(data)
        # - get pixel boundaries of thetas
        dthx = self.thx[1]-self.thx[0]
        lthx = self.thx - dthx/2.
        rthx = self.thx + dthx/2.
        dthy = self.thy[1]-self.thy[0]
        lthy = self.thy - dthy/2.
        rthy = self.thy + dthy/2.
        # - get pixel boundaries of SS
        lfD = SS.fD - SS.dfD/2.
        rfD = SS.fD + SS.dfD/2.
        ltau = SS.tau - SS.dtau/2.
        rtau = SS.tau + SS.dtau/2.
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=self.N_thx, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_thx in range(self.N_thx):
            bar.update(i_thx)
            # - compute extreme values of theta bin
            thx_l = np.min([lthx[i_thx],rthx[i_thx]])
            thx_u = np.max([lthx[i_thx],rthx[i_thx]])
            thx_2l = np.min([lthx[i_thx]**2,rthx[i_thx]**2])
            thx_2u = np.max([lthx[i_thx]**2,rthx[i_thx]**2])
            for i_thy in range(self.N_thy):
                #print(i_th1,i_th2)
                thy_l = np.min([lthy[i_thy],rthy[i_thy]])
                thy_u = np.max([lthy[i_thy],rthy[i_thy]])
                thy_2l = np.min([lthy[i_thy]**2,rthy[i_thy]**2])
                thy_2u = np.max([lthy[i_thy]**2,rthy[i_thy]**2])
                # - compute bin boundaries in SS space
                fD_u = (thx_u-thy_l)/fD_to_stau
                fD_l = (thx_l-thy_u)/fD_to_stau
                tau_u = (thx_2u-thy_2l)
                tau_l = (thx_2l-thy_2u)
                # - determine indices in SS space
                i_fD_l = int(np.rint((fD_l-SS.fD[0])/(SS.fD[-1]-SS.fD[0])*SS.N_fD))
                i_fD_u = int(np.rint((fD_u-SS.fD[0])/(SS.fD[-1]-SS.fD[0])*SS.N_fD))
                i_tau_l = int(np.rint((tau_l-SS.tau[0])/(SS.tau[-1]-SS.tau[0])*SS.N_tau))
                i_tau_u = int(np.rint((tau_u-SS.tau[0])/(SS.tau[-1]-SS.tau[0])*SS.N_tau))
                # i_fd_l = np.argmax(rfd>fd_l)
                # i_fd_u = np.argmax(rfd>fd_u)
                # i_tau_l = np.argmax(rtau>tau_l)
                # i_tau_u = np.argmax(rtau>tau_u)
                # - index 0 means out of boundaries, except for irrelevant cases
                if not (i_fD_l==0 or i_fD_u==0 or i_tau_l==0 or i_tau_u==0):
                    for i_fD in range(i_fD_l,i_fD_u+1):
                        for i_tau in range(i_tau_l,i_tau_u+1):
                            # - compute fractional area of pixel
                            area = (np.min([fD_u,rfD[i_fD]])-np.max([fD_l,lfD[i_fD]]))*(np.min([tau_u,rtau[i_tau]])-np.max([tau_l,ltau[i_tau]]))/SS.dfD/SS.dtau
                            # - read in area weighted values
                            if not (SS.fD[i_fD]==0. or SS.tau[i_tau]==0.):
                                self.thth[i_thx,i_thy] += data[i_fD,i_tau]*area
                            else:
                                self.thth[i_thx,i_thy] += noise
                            if rnoise:
                               self.thth[i_thx,i_thy] -= noise*area
        bar.finish()
        #use i_fD = int(np.rint((v_fD-fD[0])/(fD[-1]-fD[0])*N_fD)) ?
                
class thth_incoherent:
    type = "incoherent thth diagram"
    
    def __init__(self,SS,**kwargs):
        """
        self.thth: thth diagram theta*theta
        self.th: angular position
        self.th_fD: angular position in units of implied Doppler
        """
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","thth_incoherent.npz")
        if self.data_path==None:
           self.compute(SS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if SS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(SS,kwargs)
                np.savez(file_data,theta=self.th,th_fD=self.th_fD,thth=self.thth)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.th_fD = lib_data["th_fD"]
                    self.th = lib_data["theta"]
                    self.thth = lib_data["thth"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_th = len(self.th)
        
    def compute(self,SS,kwargs):
        #load and check data
        if not SS.type == "secondary spectrum":
            raise TypeError
        self.eta = kwargs.get("eta",None)
        self.Deff = kwargs.get("Deff",120.*pc)
        self.veff = kwargs.get("veff",20.*kms)
        self.nu0 = kwargs.get("nu0",1400.*MHz)
        self.N_th = kwargs.get("N_th",100)
        self.fD_max = kwargs.get("fD_max",30.*mHz)
        rnoise = kwargs.get("remove_noise",True)
        if self.eta==None:
            self.eta = v_c*self.Deff/(2.*self.nu0**2*self.veff**2)
        data = np.sqrt(SS.SS)
        
        #preparations        
        # - create containers
        self.th_fD = np.linspace(-self.fD_max,self.fD_max,num=self.N_th,endpoint=True)
        self.th = v_c/(self.nu0*self.veff)*self.th_fD
        self.thth = np.zeros((self.N_th,self.N_th),dtype=float)
        # - estimate noise
        noise = np.median(data)
        # - get pixel boundaries of thetas
        dtheta = self.th_fD[1]-self.th_fD[0]
        lthetas = self.th_fD - dtheta/2.
        rthetas = self.th_fD + dtheta/2.
        # - get pixel boundaries of SS
        lfd = SS.fD - SS.dfD/2.
        rfd = SS.fD + SS.dfD/2.
        ltau = SS.tau - SS.dtau/2.
        rtau = SS.tau + SS.dtau/2.
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=self.N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th1 in range(self.N_th):
            bar.update(i_th1)
            # - compute extreme values of theta bin
            th1_l = np.min([lthetas[i_th1],rthetas[i_th1]])
            th1_u = np.max([lthetas[i_th1],rthetas[i_th1]])
            th1_2l = np.min([lthetas[i_th1]**2,rthetas[i_th1]**2])
            th1_2u = np.max([lthetas[i_th1]**2,rthetas[i_th1]**2])
            for i_th2 in range(self.N_th):
                #print(i_th1,i_th2)
                th2_l = np.min([lthetas[i_th2],rthetas[i_th2]])
                th2_u = np.max([lthetas[i_th2],rthetas[i_th2]])
                th2_2l = np.min([lthetas[i_th2]**2,rthetas[i_th2]**2])
                th2_2u = np.max([lthetas[i_th2]**2,rthetas[i_th2]**2])
                # - compute bin boundaries in SS space
                fd_u = th1_u-th2_l
                fd_l = th1_l-th2_u
                tau_u = self.eta*(th1_2u-th2_2l)
                tau_l = self.eta*(th1_2l-th2_2u)
                # - determine indices in SS space
                i_fd_l = np.argmax(rfd>fd_l)
                i_fd_u = np.argmax(rfd>fd_u)
                i_tau_l = np.argmax(rtau>tau_l)
                i_tau_u = np.argmax(rtau>tau_u)
                # i_fd_l = int(np.rint((fd_l-SS.fD[0])/(SS.fD[-1]-SS.fD[0])*SS.N_fD))
                # i_fd_u = int(np.rint((fd_u-SS.fD[0])/(SS.fD[-1]-SS.fD[0])*SS.N_fD))
                # i_tau_l = int(np.rint((tau_l-SS.tau[0])/(SS.tau[-1]-SS.tau[0])*SS.N_tau))
                # i_tau_u = int(np.rint((tau_u-SS.tau[0])/(SS.tau[-1]-SS.tau[0])*SS.N_tau))
                # - index 0 means out of boundaries, except for irrelevant cases
                if not (i_fd_l==0 or i_fd_u==0 or i_tau_l==0 or i_tau_u==0):
                    for i_fd in range(i_fd_l,i_fd_u+1):
                        for i_tau in range(i_tau_l,i_tau_u+1):
                            # - compute fractional area of pixel
                            area = (np.min([fd_u,rfd[i_fd]])-np.max([fd_l,lfd[i_fd]]))*(np.min([tau_u,rtau[i_tau]])-np.max([tau_l,ltau[i_tau]]))/SS.dfD/SS.dtau
                            # - read in area weighted values
                            if not (SS.fD[i_fd]==0. or SS.tau[i_tau]==0.):
                                self.thth[i_th1,i_th2] += data[i_fd,i_tau]*area
                            else:
                                self.thth[i_th1,i_th2] += noise
                            if rnoise:
                               self.thth[i_th1,i_th2] -= noise*area
        bar.finish()
        #use i_fD = int(np.rint((v_fD-fD[0])/(fD[-1]-fD[0])*N_fD)) ?
        
class thth_coherent:
    type = "coherent thth diagram"
    
    def __init__(self,DS,**kwargs):
        """
        self.thth: thth diagram theta*theta
        self.th: angular position
        self.th_fD: angular position in units of implied Doppler
        """
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"thth_coherent.npz")
            if DS!=None:
                self.compute(DS,kwargs)
                np.savez(file_data,theta=self.th,th_fD=self.th_fD,thth=self.thth)
            else:
                lib_data = np.load(file_data)
                self.th_fD = lib_data["th_fD"]
                self.th = lib_data["theta"]
                self.thth = lib_data["thth"]
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_th = len(self.th)
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
            
        N_th = kwargs.get("N_th",101)
        fD_max = kwargs.get("fD_max",1.*mHz)
        npad = kwargs.get("npad",3)
        Deff = kwargs.get("Deff",120.*pc)
        veff = kwargs.get("veff",20.*kms)
        
        #preparations
        fD_to_rad = v_c/DS.nu0/veff
        self.th_fD = np.linspace(-fD_max,fD_max,num=N_th,dtype=float,endpoint=True)
        self.th = self.th_fD*fD_to_rad
        th1 = np.ones((N_th,N_th))*self.th
        th2 = th1.T
        rad_to_fD = veff*DS.nu0/v_c
        eta = v_c*Deff/(2.*DS.nu0**2*veff**2)
        self.thth = np.zeros((N_th,N_th), dtype=complex)
        
        #main computation
        dspec = DS.DS - np.mean(DS.DS)
        #Pad
        dspec_pad=np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
        #compute secondary spectrum
        SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
        fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*DS.N_t,DS.dt))
        tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*DS.N_nu,DS.dnu))
        #Compute thth diagram
        dfD = np.diff(fD).mean()
        dtau = np.diff(tau).mean()
        tau_inv = ((eta*((th1*rad_to_fD)**2-(th2*rad_to_fD)**2)-tau[0]+dtau/2)//dtau).astype(int)
        fD_inv = ((th1*rad_to_fD-th2*rad_to_fD-fD[0]+dfD/2)//dfD).astype(int)
        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fD_inv > 0) * (fD_inv < fD.shape[0])
        self.thth[pnts] = SS[fD_inv[pnts],tau_inv[pnts]]
        self.thth *= np.sqrt(np.abs(2*eta*(th2-th1))) #flux conervation
        
class thth_coherent_arc:
    type = "coherent thth diagram"
    #to do
    
    def __init__(self,DS,**kwargs):
        """
        self.thth: thth diagram theta*theta
        self.th: angular position
        self.th_fD: angular position in units of implied Doppler
        """
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
           self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"thth_coherent.npz")
            if DS!=None:
                self.compute(DS,kwargs)
                np.savez(file_data,theta=self.th,th_fD=self.th_fD,thth=self.thth)
            else:
                lib_data = np.load(file_data)
                self.th_fD = lib_data["th_fD"]
                self.th = lib_data["theta"]
                self.thth = lib_data["thth"]
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_th = len(self.th)
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
            
        N_th = kwargs.get("N_th",101)
        fD_max = kwargs.get("fD_max",1.*mHz)
        npad = kwargs.get("npad",3)
        Deff = kwargs.get("Deff",120.*pc)
        veff = kwargs.get("veff",20.*kms)
        
        #preparations
        fD_to_rad = v_c/DS.nu0/veff
        self.th_fD = np.linspace(-fD_max,fD_max,num=N_th,dtype=float,endpoint=True)
        self.th = self.th_fD*fD_to_rad
        th1 = np.ones((N_th,N_th))*self.th
        th2 = th1.T
        rad_to_fD = veff*DS.nu0/v_c
        eta = v_c*Deff/(2.*DS.nu0**2*veff**2)
        self.thth = np.zeros((N_th,N_th), dtype=complex)
        
        #main computation
        dspec = DS.DS - np.mean(DS.DS)
        #Pad
        dspec_pad=np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
        #compute secondary spectrum
        SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
        fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*DS.N_t,DS.dt))
        tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*DS.N_nu,DS.dnu))
        #Compute thth diagram
        dfD = np.diff(fD).mean()
        dtau = np.diff(tau).mean()
        tau_inv = ((eta*((th1*rad_to_fD)**2-(th2*rad_to_fD)**2)-tau[0]+dtau/2)//dtau).astype(int)
        fD_inv = ((th1*rad_to_fD-th2*rad_to_fD-fD[0]+dfD/2)//dfD).astype(int)
        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fD_inv > 0) * (fD_inv < fD.shape[0])
        self.thth[pnts] = SS[fD_inv[pnts],tau_inv[pnts]]
        self.thth *= np.sqrt(np.abs(2*eta*(th2-th1))) #flux conervation
        
class DS_backtrafo(intensity):
    type = "dynamic spectrum from backtransformation of conjugate spectrum"
    
    def __init__(self,DS,CS,**kwargs):
        """
        self.DS: dynamic spectrum time*frequency
        self.nu: frequency
        self.t: time in seconds since start of observation
        self.mjd: time in MJD
        """
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",False)
        file_name = kwargs.get("file_name","DS_backtrafo.npz")
        if self.data_path==None:
           self.compute(DS,CS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None or CS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,CS,kwargs)
                np.savez(file_data,DS=self.DS,t=self.t,nu=self.nu,mjd=self.mjd)
            else:
                lib_data = np.load(file_data)
                self.DS = lib_data["DS"]
                self.t = lib_data["t"]
                self.nu = lib_data["nu"]
                self.mjd = lib_data["mjd"]
        self.recalculate()
        
    def compute(self,DS,CS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
        if not CS.type == "conjugate spectrum":
            raise TypeError
        
        #preparations
        # - import coordinates from dynamic spectrum
        self.mjd = DS.mjd
        self.t = DS.t
        self.nu = DS.nu
        assert DS.DS.shape == CS.CS.shape
        
        #perform the computation
        # - Fourier backtransformation
        self.DS = np.real(np.fft.ifft2(np.fft.ifftshift(CS.CS,axes=(0,1)),axes=(0,1)))
        
class ConjSpec_FFT(conjugate_spectrum):
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
            if not DS.type == "intensity":
                raise TypeError
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"ConjSpec_FFT.npz")
            if DS!=None: #not os.path.exists(file_data):
                if not DS.type == "intensity":
                    raise TypeError
                self.compute(DS,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,CS=self.CS,amplitude=self.amplitude,phase=self.phase)
            else:
                lib_data = np.load(file_data)
                self.fD = lib_data["fD"]
                self.tau = lib_data["tau"]
                self.CS = lib_data["CS"]
                self.amplitude = lib_data["amplitude"]
                self.phase = lib_data["phase"]
        self.recalculate()
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
        
        #perform the computation
        # - Fourier transformation
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        self.CS = np.fft.fftshift(np.fft.fft2(DS.DS,axes=(0,1)),axes=(0,1))
        self.phase = np.angle(self.CS)
        self.amplitude = np.abs(self.CS)
        
class IntensityCrossSpectrum(conjugate_spectrum):
    def __init__(self,CS1,CS2,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        if self.data_path==None:
            self.compute(CS1,CS2,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"IntensityCrossSpectrum.npz")
            if CS1!=None or CS2!=None:
                self.compute(CS1,CS2,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,ICS=self.CS,amplitude=self.amplitude,phase=self.phase)
            else:
                lib_data = np.load(file_data)
                self.fD = lib_data["fD"]
                self.tau = lib_data["tau"]
                self.CS = lib_data["CS"]
                self.amplitude = lib_data["amplitude"]
                self.phase = lib_data["phase"]
        self.recalculate()
        
    def compute(self,CS1,CS2,kwargs):
        #load and check data
        if not CS1.type == "conjugate spectrum" and CS2.type == "conjugate spectrum":
            raise TypeError
            
        #compute the cross-correlation
        A = CS1.CS
        B = np.flip(CS2.CS)
        ixA = np.argmin(np.abs(CS1.fD))
        ixB = np.argmin(np.abs(np.flip(CS2.fD)))
        iyA = np.argmin(np.abs(CS1.tau))
        iyB = np.argmin(np.abs(np.flip(CS2.tau)))
        B = np.roll(B,(ixA-ixB,iyA-iyB),axis=(0,1))
        
        self.fD = CS1.fD
        self.tau = CS1.tau
        self.CS = A*B
        self.amplitude = np.abs(self.CS)
        self.phase = np.angle(self.CS)
        
class eigenvectors:
    type = "eigenvectors"
    
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","eigenvectors.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,th=self.th,t=self.t,nu=self.nu,eigenvectors=self.eigenvectors)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.th = lib_data["th"]
                    self.t = lib_data["t"]
                    self.nu = lib_data["nu"]
                    self.eigenvectors = lib_data["eigenvectors"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_th = len(self.th)
        self.N_t = len(self.t)
        self.N_nu = len(self.nu)
        self.timespan = self.t[-1] - self.t[0]
        self.t0 = np.mean(self.t)
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
            
        N_th = kwargs.get("N_th",100)
        fD_max = kwargs.get("fD_max",1.*mHz)
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        npad = kwargs.get("npad",3)
        Deff = kwargs.get("Deff",120.*pc)
        veff = kwargs.get("veff",20.*kms)
        #self.eta = v_c*self.Deff/(2.*self.nu0**2*self.veff**2)
        
        #preparations
        fD_to_rad = v_c/DS.nu0/veff
        self.th = np.linspace(-fD_max*fD_to_rad,fD_max*fD_to_rad,num=N_th,dtype=float,endpoint=True)
        N_nuchunk = int(DS.N_nu/nuchunk)
        self.nu = np.zeros(N_nuchunk,dtype=float)
        N_tchunk = int(DS.N_t/tchunk)
        self.t = np.zeros(N_tchunk,dtype=float)
        self.eigenvectors = np.empty((N_th,N_tchunk,N_nuchunk),dtype=complex)
        th1 = np.ones((N_th,N_th))*self.th
        th2 = th1.T
        
        #main computation
        bar = progressbar.ProgressBar(maxval=N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for fc in range(N_nuchunk):
            bar.update(fc)
            #select Chunk and determine curvature
            fc0 = fc*nuchunk
            fc1 = fc0+nuchunk
            self.nu[fc] = np.mean(DS.nu[fc0:fc1])
            rad_to_fD = veff*self.nu[fc]/v_c
            eta = v_c*Deff/(2.*self.nu[fc]**2*veff**2)
            for tc in range(N_tchunk):
                tc0 = tc*tchunk
                tc1 = tc0+tchunk
                self.t[tc] = np.mean(DS.t[tc0:tc1])
                dspec = DS.DS[tc0:tc1,fc0:fc1]
                dspec = dspec - np.mean(dspec)
                #Pad
                dspec_pad=np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
                #compute secondary spectrum
                SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*tchunk,DS.dt))
                tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nuchunk,DS.dnu))
                #Compute thth diagram
                dfD = np.diff(fD).mean()
                dtau = np.diff(tau).mean()
                tau_inv = ((eta*((th1*rad_to_fD)**2-(th2*rad_to_fD)**2)-tau[0]+dtau/2)//dtau).astype(int)
                fD_inv = ((th1*rad_to_fD-th2*rad_to_fD-fD[0]+dfD/2)//dfD).astype(int)
                thth = np.zeros((N_th,N_th), dtype=complex)
                pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fD_inv > 0) * (fD_inv < fD.shape[0])
                thth[pnts] = SS[fD_inv[pnts],tau_inv[pnts]]
                thth *= np.sqrt(np.abs(2*eta*(th2-th1))) #flux conervation
                thth -= np.tril(thth) #make hermitian
                thth += np.conjugate(np.triu(thth).T)
                thth -= np.diag(np.diag(thth))
                thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                thth = np.nan_to_num(thth)
                #Compute dominant eigenvector
                w,V = eigsh(thth,1)
                w = np.abs(w[0]) #added abs to get valid square roots
                V = V[:,0]
                solution = np.conjugate(V)*np.sqrt(w)*N_th
                self.eigenvectors[:,tc,fc] = np.nan_to_num(solution)
        bar.finish()
        
class brightness_dist:
    type = "brightness distribution"
        
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","mu.npz")
        self.file_name = file_name
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,stau=self.stau,t=self.t,nu=self.nu,mu=self.mu)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.stau = lib_data["stau"]
                    self.t = lib_data["t"]
                    self.nu = lib_data["nu"]
                    self.mu = lib_data["mu"]
                else:
                    raise KeyError
        self.recalculate()
        
    def crop(self,t_min=None,t_max=None,nu_min=None,nu_max=None,N_t=None,N_nu=None): #missing option for mask, profile and bpass
            # - create subset of data
        if t_min==None:
            i0_t = 0
        else:
            i0_t = np.argmin(np.abs(self.t-t_min))
        if t_max==None:
            i1_t = self.N_t
            if not N_t==None:
                if N_t <= i1_t:
                    i1_t = N_t
                else:
                    print("/!\ N_t too large! Using available data instead.")
        else:
            i1_t = np.argmin(np.abs(self.t-t_max))
            if not N_t==None and N_t!=i1_t:
                print("/!\ N_t incompatible with t_max! Using only t_max instead.")
        if nu_min==None:
            i0_nu = 0
        else:
            i0_nu = np.argmin(np.abs(self.nu-nu_min))
        if nu_max==None:
            i1_nu = self.N_nu
            if not N_nu==None:
                if N_nu <= i1_nu:
                    i1_nu = N_nu
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_nu = np.argmin(np.abs(self.nu-nu_max))
            if not N_nu==None and N_nu!=i1_nu:
                print("/!\ N_nu incompatible with nu_max! Using only nu_max instead.")
        if i0_t!=0 or i1_t!=self.N_t or i0_nu!=0 or i1_nu!=self.N_nu:
            self.t = self.t[i0_t:i1_t]
            self.nu = self.nu[i0_nu:i1_nu]
            self.mu = self.mu[:,i0_t:i1_t,i0_nu:i1_nu]
            self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_stau = len(self.stau)
        self.N_th = self.N_stau
        self.N_t = len(self.t)
        self.N_nu = len(self.nu)
        self.timespan = self.t[-1] - self.t[0]
        self.t0 = np.mean(self.t)
        self.dstau = self.stau[1]-self.stau[0]
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
        self.nu0 = DS.nu0
        method = kwargs.get("method","eigenvector")
        if method=="eigenvector":
            self.compute_eigenvector(DS,kwargs)
        elif method=="backtrafo":
            self.compute_backtrafo(DS,kwargs)
        elif method=="rotated":
            self.stau = np.empty(2)
            self.t = np.empty(2)
            self.nu = np.empty(2)
            self.mu = np.empty(2)
            
    def save_eigenvectors(self):
        file_data = os.path.join(self.data_path,self.file_name)
        np.savez(file_data,stau=self.stau,t=self.t,nu=self.nu,mu=self.mu)
            
    def compute_eigenvector(self,DS,kwargs):
        N_th = kwargs.get("N_th",100)
        tau_max = kwargs.get("tau_max",1.*musec)
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        npad = kwargs.get("npad",3)
        self.eta1400 = kwargs.get("eta",0.1) #at 1400 MHz
        
        #preparations
        stau_max = np.sqrt(tau_max)
        self.stau = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
        N_nuchunk = int(DS.N_nu/nuchunk)
        self.nu = np.zeros(N_nuchunk,dtype=float)
        N_tchunk = int(DS.N_t/tchunk)
        self.t = np.zeros(N_tchunk,dtype=float)
        self.mu = np.empty((N_th,N_tchunk,N_nuchunk),dtype=complex)
        th1 = np.ones((N_th,N_th))*self.stau
        th2 = th1.T
        
        #main computation
        bar = progressbar.ProgressBar(maxval=N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for fc in range(N_nuchunk):
            bar.update(fc)
            #select Chunk and determine curvature
            fc0 = fc*nuchunk
            fc1 = fc0+nuchunk
            self.nu[fc] = np.mean(DS.nu[fc0:fc1])
            eta = self.eta1400*(1400.0*MHz)**2/self.nu[fc]**2
            stau_to_fD = 1./np.sqrt(eta)
            for tc in range(N_tchunk):
                tc0 = tc*tchunk
                tc1 = tc0+tchunk
                self.t[tc] = np.mean(DS.t[tc0:tc1])
                dspec = DS.DS[tc0:tc1,fc0:fc1]
                dspec = dspec - np.mean(dspec)
                #Pad
                dspec_pad=np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
                #compute secondary spectrum
                SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*tchunk,DS.dt))
                tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nuchunk,DS.dnu))
                #Compute thth diagram
                dfD = np.diff(fD).mean()
                dtau = np.diff(tau).mean()
                tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                fD_inv = ((th1*stau_to_fD-th2*stau_to_fD-fD[0]+dfD/2)//dfD).astype(int)
                thth = np.zeros((N_th,N_th), dtype=complex)
                pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fD_inv > 0) * (fD_inv < fD.shape[0])
                thth[pnts] = SS[fD_inv[pnts],tau_inv[pnts]]
                #thth *= np.sqrt(np.abs(2*eta*(th2-th1))) #flux conervation
                thth *= np.abs(2*eta*(th2-th1)) #Jacobian
                thth -= np.tril(thth) #make hermitian
                thth += np.conjugate(np.triu(thth).T)
                thth -= np.diag(np.diag(thth))
                thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                thth = np.nan_to_num(thth)
                #Compute dominant eigenvector
                w,V = eigsh(thth,1)
                w = np.abs(w[0]) #added abs to get valid square roots
                V = V[:,0]
                solution = np.conjugate(V)*np.sqrt(w)*N_th
                self.mu[:,tc,fc] = np.nan_to_num(solution)
        bar.finish()
        
    def invert_modulation(self,**kwargs):
        
        method = kwargs.get("method","eigenvector")
        recompute = kwargs.get("recompute",True)
        subfile = "DS_nomod.npz"
        file_data = os.path.join(self.data_path,subfile)
        if recompute:
            if method=="eigenvector":
                N_th,N_tchunk,N_nuchunk = self.mu.shape
                tchunk = kwargs.get("tchunk",50)
                nuchunk = kwargs.get("nuchunk",300)
                npad = kwargs.get("npad",3)
                self.eta1400 = kwargs.get("eta",0.1) #at 1400 MHz
                
                #preparations
                stau_max = self.stau[-1]
                tau_max = stau_max**2
                #self.stau = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
                #self.nu = np.zeros(N_nuchunk,dtype=float)
                #self.t = np.zeros(N_tchunk,dtype=float)
                #self.mu = np.empty((N_th,N_tchunk,N_nuchunk),dtype=complex)
                
                N_t = int(N_tchunk*tchunk)
                N_nu = int(N_nuchunk*nuchunk)
                dspec_model = np.empty((N_t,N_nu),dtype=float)
                dt = (self.t[1]-self.t[0])/tchunk
                t_model = np.arange(N_t)*dt
                dnu = (self.nu[1]-self.nu[0])/nuchunk
                nu_model = np.arange(N_nu)*dnu
                
                modulation = np.mean(np.abs(self.mu),axis=2)
                mean_images = np.mean(np.abs(self.mu),axis=(1,2))
                modulation_normed = modulation / mean_images[:,na]
                mu_clean = self.mu/modulation_normed[:,:,na]
                
                fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*tchunk,dt))
                tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nuchunk,dnu))
                dfD = np.diff(fD).mean()
                dtau = np.diff(tau).mean()
                fD_edges = np.linspace(fD[0]-dfD/2.,fD[-1]+dfD/2.,fD.shape[0]+1,endpoint=True)
                tau_edges = np.linspace(tau[0]-dtau/2.,tau[-1]+dtau/2.,tau.shape[0]+1,endpoint=True)
                
                #main computation
                bar = progressbar.ProgressBar(maxval=N_nuchunk*N_tchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for fc in range(N_nuchunk):
                    fc0 = fc*nuchunk
                    fc1 = fc0+nuchunk
                    #select Chunk and determine curvature
                    eta = self.eta1400*(1400.0*MHz)**2/self.nu[fc]**2
                    stau_to_fD = 1./np.sqrt(eta)
                    fD_map = (self.stau[na,:]-self.stau[:,na])*stau_to_fD
                    tau_map = (self.stau[na,:]**2-self.stau[:,na]**2)
                    for tc in range(N_tchunk):
                        bar.update(tc+fc*N_tchunk)
                        tc0 = tc*tchunk
                        tc1 = tc0+tchunk
                        #Construct 1D theta-theta
                        thth1D = np.zeros((N_th,N_th), dtype=complex)
                        thth1D[thth1D.shape[0]//2,:] = mu_clean[:,tc,fc]
                        with np.errstate(all='ignore'):
                            recov=np.histogram2d(np.ravel(fD_map),
                                         np.ravel(tau_map),
                                         bins=(fD_edges,tau_edges),
                                         weights=np.ravel(thth1D/np.sqrt(np.abs(2*eta*fD_map.T))).real)[0] +\
                                    np.histogram2d(np.ravel(fD_map),
                                                 np.ravel(tau_map),
                                                 bins=(fD_edges,tau_edges),
                                                 weights=np.ravel(thth1D/np.sqrt(np.abs(2*eta*fD_map.T))).imag)[0]*1j
                            norm=np.histogram2d(np.ravel(fD_map),
                                                 np.ravel(tau_map),
                                                 bins=(fD_edges,tau_edges))[0]
                            recov /= norm
                        recov = np.nan_to_num(recov)
                        model_E = np.fft.ifft2(np.fft.ifftshift(recov))[:tchunk,:nuchunk]
                        dspec_model[tc0:tc1,fc0:fc1] = np.abs(model_E)**2
                bar.finish()
                
                np.savez(file_data,t_model=t_model,nu_model=nu_model,dspec_model=dspec_model)
        else:
            lib_data = np.load(file_data)
            t_model = lib_data["t_model"]
            nu_model = lib_data["nu_model"]
            dspec_model = lib_data["dspec_model"]
            
        return t_model,nu_model,dspec_model
        
    def compute_backtrafo(self,DS,kwargs):
    
        self.N_th = kwargs.get("N_th",100)
        tau_max = kwargs.get("tau_max",1.*musec)
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        npad = kwargs.get("npad",3)
        self.eta1400 = kwargs.get("eta",0.1)
        fD_width_inner = kwargs.get("fD_width_inner",1.)
        fD_width_outer = kwargs.get("fD_width_outer",2.)
        tau_width_inner = kwargs.get("tau_width_inner",1.)
        tau_width_outer = kwargs.get("tau_width_outer",2.)
        tau_center = kwargs.get("tau_center",1.)
        fD_center = kwargs.get("fD_center",1.)
        stripe = kwargs.get("stripe",False)
        
        #preparations
        # - Fourier transformation
        fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        CS = np.fft.fftshift(np.fft.fft2(DS.DS,axes=(0,1)),axes=(0,1))
        for i_fD,v_fD in enumerate(fD):
            if np.abs(v_fD)<fD_center:
                CS[i_fD,:] = 0.
            else:
                for i_tau,v_tau in enumerate(tau):
                    if np.abs(v_tau)<tau_center:
                        CS[i_fD,i_tau] = 0.
        N_fD,N_tau = CS.shape
        dfD = np.diff(fD).mean()
        dtau = np.diff(tau).mean()
        # - downsampling
        coordinates = np.array([DS.t,DS.t])
        coordinates = block_reduce(coordinates, block_size=(1,tchunk), func=np.mean, cval=DS.t[-1])
        self.t = coordinates[0,:]
        coordinates = np.array([DS.nu,DS.nu])
        coordinates = block_reduce(coordinates, block_size=(1,nuchunk), func=np.mean, cval=DS.nu[-1])
        self.nu = coordinates[0,:]
        self.N_t = len(self.t)
        self.N_nu = len(self.nu)
        # - theta scaling
        stau_max = np.sqrt(tau_max)
        self.stau = np.linspace(-stau_max,stau_max,num=self.N_th,dtype=float,endpoint=True)
        self.mu = np.empty((self.N_th,self.N_t,self.N_nu),dtype=float)
        eta = self.eta1400*(1400.0*MHz)**2/DS.nu0**2
        stau_to_fD = 1./np.sqrt(eta)
        # - define mask
        fD_freq = np.pi/2./(fD_width_outer-fD_width_inner)
        tau_freq = np.pi/2./(tau_width_outer-tau_width_inner)
        def mask(dfD,dtau):
            if dfD>fD_width_outer:
                factor = 0.
            elif dfD>fD_width_inner:
                factor = np.cos(fD_freq*(dfD-fD_width_inner))**2
            else:
                factor = 1.
            if dtau>tau_width_outer:
                factor *= 0.
            elif dtau>tau_width_inner:
                factor *= np.cos(tau_freq*(dtau-tau_width_inner))**2
            return factor
        arr = np.zeros((N_fD,N_tau))
        for i_fD,v_fD in enumerate(fD):
            if np.abs(v_fD)<fD_width_outer:
                for i_tau,v_tau in enumerate(tau):
                    if np.abs(v_tau)<tau_width_outer:
                        arr[i_fD,i_tau] = mask(np.abs(v_fD),np.abs(v_tau))
        if not stripe:
            for i_fD,v_fD in enumerate(fD):
                col = arr[i_fD,:]
                shift = int(np.rint(eta*v_fD**2/dtau))
                col = np.roll(col,shift)
                arr[i_fD,:] = col
        
        #main computation
        bar = progressbar.ProgressBar(maxval=self.N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in range(self.N_th):
            bar.update(i_th)
            fD0 = self.stau[i_th]*stau_to_fD
            tau0 = self.stau[i_th]**2
            di_fD = int(np.rint(fD0/dfD))
            di_tau = int(np.rint(tau0/dtau))
            f_mask = np.roll(arr,(di_fD,di_tau),axis=(0,1)) + np.roll(arr,(-di_fD,-di_tau),axis=(0,1))
            f_mask[f_mask>1.] = 1.
            source = np.fft.ifftshift(CS*f_mask,axes=(0,1))
            # - Fourier backtransformation
            result = np.abs(np.fft.ifft2(source,axes=(0,1)))
            self.mu[i_th,:,:] = block_reduce(result, block_size=(tchunk,nuchunk), func=np.mean)
        bar.finish()
         
        
    def fit_modulation_speed(self,**kwargs):
        a_min = kwargs.get("a_min",1.)
        a_max = kwargs.get("a_max",3.)
        threshold_y = kwargs.get("threshold_y",0.0)
        threshold_x = kwargs.get("threshold_x",np.inf)
        threshold_stau = kwargs.get("threshold_stau",0.)
        fraction = kwargs.get("fraction",0.5)
        load_from = kwargs.get("load_from",None)
        
        recompute = True
    
        if load_from != None:
            savefile = os.path.join(self.data_path,load_from)
            if os.path.exists(savefile):
                recompute = False
            
        if recompute:
            #modulation = np.swapaxes(np.mean(np.abs(self.mu),axis=2)**2,0,1)
            #modulation_normed = modulation / np.mean(np.abs(self.mu),axis=(1,2))[na,:]**2
            modulation = np.swapaxes(np.mean(np.abs(self.mu),axis=2),0,1)
            modulation_normed = modulation / np.mean(np.abs(self.mu),axis=(1,2))[na,:]
            
            N_a = int((self.t[-1]-self.t[0])*(a_max-a_min)/self.dstau)
            a = np.linspace(a_min,a_max,num=N_a,endpoint=True)
            Hough = np.zeros((N_a,self.N_th),dtype=float)
            
            #perform the computation
            weights = np.zeros((N_a,self.N_th),dtype=int)
            bar = progressbar.ProgressBar(maxval=N_a*self.N_t , widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i_a in range(N_a):
                for i_t in range(self.N_t):
                    bar.update(i_a*self.N_t+i_t)
                    for i_th0 in range(self.N_th):
                        i_th = i_th0 + int(np.rint((self.t[i_t]-self.t[0])*a[i_a]/self.dstau))
                        if 0<=i_th<self.N_th:
                            if not np.abs(self.stau[i_th])<threshold_stau:
                                Hough[i_a,i_th0] += modulation_normed[i_t,i_th]
                                weights[i_a,i_th0] += 1
            bar.finish()
            weights[weights==0] = 1
            Hough = Hough/weights
            #filler = np.mean(Hough)
            Hough[weights<fraction*self.N_t] = np.nan
            measure = np.nanstd(Hough,axis=1)
            
            # - fit peak
            threshold = threshold_y*np.max(measure[a<threshold_x])
            mask = np.where((measure>threshold) & (a<threshold_x))[0]
            xdata = a[mask]
            ydata = measure[mask]
            xscale = np.mean(xdata)
            yscale = np.mean(ydata)
            cscale = np.abs((np.min(ydata)-np.max(ydata))/4./(xdata[-1]-xdata[0])**2)
            def parabola(data,x_in,y_in,curv_in):
                x = x_in*xscale
                y = y_in*yscale
                curv = curv_in*cscale
                #print(x,y,curv)
                return curv*(data-x)**2+y
            fit_stau = xdata
            try:
                popt, pcov = curve_fit(parabola,xdata,ydata,p0=[np.mean(xdata)/xscale,np.max(ydata)/yscale,-1.],bounds=([np.min(xdata)/xscale,np.min(ydata)/yscale,-np.inf],[np.max(xdata)/xscale,np.max(ydata)/yscale,0.]))
                perr = np.sqrt(np.diag(pcov))
                fit_result = popt
                fit_error = perr
                fit_curve = parabola(xdata,fit_result[0],fit_result[1],fit_result[2])
            except RuntimeError:
                print("Error - curve_fit failed")
                fit_curve = np.zeros_like(xdata)
                fit_result = np.full(3,np.nan)
                fit_error = np.full(3,np.nan)
            vm_result = fit_result[0]*xscale
            vm_error = fit_error[0]*xscale
            print("Modulation speed fit peak: {0} +- {1}".format(vm_result,vm_error))
            
            if load_from != None:
                np.savez(savefile,modulation=modulation,modulation_normed=modulation_normed,a=a,Hough=Hough,fit_stau=fit_stau,fit_curve=fit_curve,measure=measure,vm_result=vm_result,vm_error=vm_error)
        else:
            lib = np.load(savefile)
            modulation = lib["modulation"]
            modulation_normed = lib["modulation_normed"]
            a = lib["a"]
            Hough = lib["Hough"]
            fit_stau = lib["fit_stau"]
            fit_curve = lib["fit_curve"]
            measure = lib["measure"]
            vm_result = lib["vm_result"]
            vm_error = lib["vm_error"]
        
        return modulation,modulation_normed,a,Hough,fit_stau,fit_curve,measure,vm_result,vm_error
        
        
class Efield:
    type = "electric field"
    
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","Efield.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,t=self.t,nu=self.nu,Efield=self.Efield,amplitude=self.amplitude,phase=self.phase)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.t = lib_data["t"]
                    self.nu = lib_data["nu"]
                    self.Efield = lib_data["Efield"]
                    self.amplitude = lib_data["amplitude"]
                    self.phase = lib_data["phase"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t = len(self.t)
        self.N_nu = len(self.nu)
        
    def crop(self,t_min=None,t_max=None,nu_min=None,nu_max=None,N_t=None,N_nu=None): #missing option for mask, profile and bpass
            # - create subset of data
        if t_min==None:
            i0_t = 0
        else:
            i0_t = np.argmin(np.abs(self.t-t_min))
        if t_max==None:
            i1_t = self.N_t
            if not N_t==None:
                if N_t <= i1_t:
                    i1_t = N_t
                else:
                    print("/!\ N_t too large! Using available data instead.")
        else:
            i1_t = np.argmin(np.abs(self.t-t_max))
            if not N_t==None and N_t!=i1_t:
                print("/!\ N_t incompatible with t_max! Using only t_max instead.")
        if nu_min==None:
            i0_nu = 0
        else:
            i0_nu = np.argmin(np.abs(self.nu-nu_min))
        if nu_max==None:
            i1_nu = self.N_nu
            if not N_nu==None:
                if N_nu <= i1_nu:
                    i1_nu = N_nu
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_nu = np.argmin(np.abs(self.nu-nu_max))
            if not N_nu==None and N_nu!=i1_nu:
                print("/!\ N_nu incompatible with nu_max! Using only nu_max instead.")
        if i0_t!=0 or i1_t!=self.N_t or i0_nu!=0 or i1_nu!=self.N_nu:
            self.t = self.t[i0_t:i1_t]
            self.nu = self.nu[i0_nu:i1_nu]
            self.Efield = self.Efield[i0_t:i1_t,i0_nu:i1_nu]
            self.amplitude = self.amplitude[i0_t:i1_t,i0_nu:i1_nu]
            self.phase = self.phase[i0_t:i1_t,i0_nu:i1_nu]
            self.recalculate()
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
            
        N_th = kwargs.get("N_th",100)
        #fD_max = kwargs.get("fD_max",1.*mHz)
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        npad = kwargs.get("npad",3)
        #Deff = kwargs.get("Deff",120.*pc)
        #veff = kwargs.get("veff",20.*kms)
        tau_max = kwargs.get("tau_max",1.*mus)
        zeta = kwargs.get("zeta",1.)
        
        mus_object = kwargs.get("provide_mus",None)
        
        def find_chunks(N,Nc):
            """
            N : length of list
            Nc : Maximum number of entries per chunk
            """
            #number of chunks
            Ns = (2*N)//Nc
            #optimal shift of chunk (float)
            shift = N/(Ns+1.)
            #create list by rounding to integers
            starts = [np.rint(i*shift).astype(int) for i in range(Ns)]
            #mids = [np.rint((i+1)*shift).astype(int) for i in range(Ns)]
            ends = [np.rint((i+2)*shift).astype(int) for i in range(Ns)]
            return starts,ends,int(shift)
        
        #preparations
        #fD_to_rad = v_c/DS.nu0/veff
        #self.th = np.linspace(-fD_max*fD_to_rad,fD_max*fD_to_rad,num=N_th,dtype=float,endpoint=True)
        #th1 = np.ones((N_th,N_th))*self.th
        #th2 = th1.T
        
        stau_max = np.sqrt(tau_max)
        staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
        th1 = np.ones((N_th,N_th))*staus
        th2 = th1.T
        
        self.Efield = np.zeros(DS.DS.shape,dtype=complex)
        #- split into overlapping chunks
        t_starts,t_ends,t_shift = find_chunks(DS.N_t,tchunk)
        nu_starts,nu_ends,nu_shift = find_chunks(DS.N_nu,nuchunk)
        N_tchunk = len(t_starts)
        N_nuchunk = len(nu_starts)
        # #Compute size and number of chunks
        # hnuchunk = int(nuchunk/2)
        # htchunk = int(tchunk/2)
        # N_tchunk=(DS.N_t-htchunk)//htchunk
        # N_nuchunk=(DS.N_nu-hnuchunk)//hnuchunk
        
        save_mus = False
        if mus_object != None:
            save_mus = True
            mu = np.empty((N_th,N_tchunk,N_nuchunk),dtype=complex)
            ts = np.empty(N_tchunk,dtype=float)
            nus = np.empty(N_nuchunk,dtype=float)
        
        ###reference from similar code:
        # stau_max = np.sqrt(tau_max)
        # staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
        # th1 = np.ones((N_th,N_th))*staus
        # stau_to_fD = 2.*f0_evo*zeta
        # tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
        # fd_inv = ((stau_to_fD*(th1-th2)-fd[0]+dfd/2)//dfd).astype(int)
        # eta = 1./(2.*f0_evo*zeta)**2
        # thth *= np.sqrt(np.abs(2*eta*stau_to_fD*(th2-th1))) #flux conervation
            
        #main computation
        bar = progressbar.ProgressBar(maxval=N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for fc in range(N_nuchunk):
            bar.update(fc)
            #select Chunk and determine curvature
            l_nuchunk = nu_ends[fc]-nu_starts[fc]
            nu = DS.nu[nu_starts[fc]:nu_ends[fc]]
            nu0 = np.mean(nu)
            #rad_to_fD = veff*nu0/v_c
            #eta = v_c*Deff/(2.*nu0**2*veff**2)
            stau_to_fD = 2.*nu0*zeta
            eta = 1./(2.*nu0*zeta)**2
            #Map back to time/frequency space
            fD_map = (staus[na,:]-staus[:,na])*stau_to_fD
            tau_map = (staus[na,:]**2-staus[:,na]**2)
            # combine by windowing
            fmsk = np.ones(l_nuchunk)
            if fc>0:
                fmsk[:nu_shift] = np.sin((np.pi/2)*np.linspace(0,nu_shift-1,nu_shift)/nu_shift)**2
            if fc<N_nuchunk-1:
                fmsk[-nu_shift:] = np.cos((np.pi/2)*np.linspace(0,nu_shift-1,nu_shift)/nu_shift)**2
            for tc in range(N_tchunk):
                t = DS.t[t_starts[tc]:t_ends[tc]]
                l_tchunk = t_ends[tc]-t_starts[tc]
                tmsk = np.ones(l_tchunk)
                dspec = DS.DS[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]]
                dspec = dspec - np.mean(dspec)
                #Pad
                dspec_pad = np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())
                #compute secondary spectrum
                SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                fD = np.fft.fftshift(np.fft.fftfreq((npad+1)*t.shape[0],t[1]-t[0]))
                tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nu.shape[0],nu[1]-nu[0]))
                dfD = np.diff(fD).mean()
                dtau = np.diff(tau).mean()
                fD_edges = np.linspace(fD[0]-dfD/2.,fD[-1]+dfD/2.,fD.shape[0]+1,endpoint=True)
                tau_edges = np.linspace(tau[0]-dtau/2.,tau[-1]+dtau/2.,tau.shape[0]+1,endpoint=True)
                #Compute thth diagram
                #tau_inv = ((eta*((th1*rad_to_fD)**2-(th2*rad_to_fD)**2)-tau[0]+dtau/2)//dtau).astype(int)
                #fD_inv = ((th1*rad_to_fD-th2*rad_to_fD-fD[0]+dfD/2)//dfD).astype(int)
                tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                fD_inv = ((th1*stau_to_fD-th2*stau_to_fD-fD[0]+dfD/2)//dfD).astype(int)
                thth = np.zeros((N_th,N_th), dtype=complex)
                pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fD_inv > 0) * (fD_inv < fD.shape[0])
                thth[pnts] = SS[fD_inv[pnts],tau_inv[pnts]]
                #thth *= np.sqrt(np.abs(2*eta*(th2-th1))) #flux conervation
                thth *= np.abs(2*eta*(th2-th1)) #Jacobian
                thth -= np.tril(thth) #make hermitian
                thth += np.conjugate(np.triu(thth).T)
                thth -= np.diag(np.diag(thth))
                thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                thth = np.nan_to_num(thth)
                #Compute dominant eigenvector
                w,V = eigsh(thth,1)
                w = np.abs(w[0]) #added abs to get valid square roots
                V = V[:,0]
                eigenvectors = np.conjugate(V)*np.sqrt(w)*N_th
                eigenvectors = np.nan_to_num(eigenvectors)
                #Construct 1D theta-theta
                thth1D = np.zeros((N_th,N_th), dtype=complex)
                thth1D[thth1D.shape[0]//2,:] = eigenvectors
                with np.errstate(all='ignore'):
                    recov=np.histogram2d(np.ravel(fD_map),
                                 np.ravel(tau_map),
                                 bins=(fD_edges,tau_edges),
                                 weights=np.ravel(thth1D/np.sqrt(np.abs(2*eta*fD_map.T))).real)[0] +\
                            np.histogram2d(np.ravel(fD_map),
                                         np.ravel(tau_map),
                                         bins=(fD_edges,tau_edges),
                                         weights=np.ravel(thth1D/np.sqrt(np.abs(2*eta*fD_map.T))).imag)[0]*1j
                    norm=np.histogram2d(np.ravel(fD_map),
                                         np.ravel(tau_map),
                                         bins=(fD_edges,tau_edges))[0]
                    recov /= norm
                recov = np.nan_to_num(recov)
                model_E = np.fft.ifft2(np.fft.ifftshift(recov))[:dspec.shape[0],:dspec.shape[1]]
                #Combine chunks
                old = self.Efield[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]]
                new = model_E
                old_amps = np.abs(old)
                new_amps = np.abs(new)
                diff = old*np.conj(new)
                phi = np.angle(np.sum(diff))
                model_E = model_E*np.exp(1.j*phi)
                tmsk = np.ones(l_tchunk)
                if tc>0:
                    tmsk[:t_shift] = np.sin((np.pi/2)*np.linspace(0,t_shift-1,t_shift)/t_shift)**2
                if tc<N_tchunk-1:
                    tmsk[-t_shift:] = np.cos((np.pi/2)*np.linspace(0,t_shift-1,t_shift)/t_shift)**2
                self.Efield[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]] += model_E*tmsk[:,na]*fmsk[na,:]
                
                if save_mus:
                    nus[fc] = nu0
                    ts[tc] = np.mean(t)
                    mu[:,tc,fc] = eigenvectors*np.exp(1.j*phi)
        bar.finish()
        
        self.t = DS.t
        self.nu = DS.nu
        self.amplitude = np.abs(self.Efield)
        self.phase = np.angle(self.Efield)
        
        if save_mus:
            mus_object.stau = staus
            mus_object.t = ts
            mus_object.nu = nus
            mus_object.mu = mu
            mus_object.save_eigenvectors()
        
class generic_Efield(Efield):
    def __init__(self,t,nu,EF):
        self.t = t
        self.nu = nu
        self.Efield = EF
        self.amplitude = np.abs(EF)
        self.phase = np.angle(EF)
        self.recalculate()
    
class load_Efield(Efield):    
    def __init__(self,data_path):
        self.data_path = data_path
        file_data = os.path.join(data_path,"EF.npz")
        lib_data = np.load(file_data)
        self.nu = lib_data["nu"]
        self.t = lib_data["t"]
        self.Efield = lib_data["Efield"]
        self.amplitude = np.abs(self.Efield)
        self.phase = np.angle(self.Efield)
        self.recalculate()
        
class Wavefield_NuT:
    type = "wavefield"
    def __init__(self,DS,EF,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","Wavefield_NuT.npz")
        if self.data_path==None:
            self.compute(DS,EF,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None or EF==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,EF,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,WF=self.WF,nu0=self.nu0,amplitude=self.amplitude,phase=self.phase)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.tau = lib_data["tau"]
                    self.WF = lib_data["WF"]
                    self.nu0 = lib_data["nu0"]
                    self.amplitude = lib_data["amplitude"]
                    self.phase = lib_data["phase"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_fD,self.N_tau = self.WF.shape
        self.dfD = self.fD[1] - self.fD[0]
        self.dtau = self.tau[1] - self.tau[0]
    
    def compute(self,DS,EF,kwargs):
        #print("Computing Wavefield_NuT ...")
        if not DS.type == "intensity":
            raise TypeError
        if not EF.type == "electric field":
            raise 
        self.nu0 = kwargs.get("nu0",DS.nu0)
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        self.N_fD = len(self.fD)
        self.N_tau = len(self.tau)
        
        #- prepare data
        data = np.copy(DS.DS)
        data[data<0.] = 0.
        E = np.sqrt(data)*np.exp(1.j*EF.phase)
        tt = (DS.t-DS.t0)/self.nu0
        hWF_real = np.zeros((self.N_fD*EF.N_nu),dtype='float64')
        hWF_im = np.zeros((self.N_fD*EF.N_nu),dtype='float64')
        lib.ENuT(EF.N_t,EF.N_nu,self.N_fD,tt,EF.nu,self.fD,np.real(E).flatten(),np.imag(E).flatten(),hWF_real,hWF_im)
        hWF = hWF_real.reshape((self.N_fD,EF.N_nu))+1.j*hWF_im.reshape((self.N_fD,EF.N_nu))
        self.WF = np.fft.fftshift(np.fft.fft(hWF,axis=1),axes=1)
        
        self.amplitude = np.abs(self.WF)
        self.phase = np.angle(self.WF)
        
class Wavefield_FFT(Wavefield_NuT):
    def __init__(self,DS,EF,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","Wavefield_FFT.npz")
        if self.data_path==None:
            self.compute(DS,EF,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None or EF==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,EF,kwargs)
                np.savez(file_data,fD=self.fD,tau=self.tau,WF=self.WF,nu0=self.nu0,amplitude=self.amplitude,phase=self.phase)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.fD = lib_data["fD"]
                    self.tau = lib_data["tau"]
                    self.WF = lib_data["WF"]
                    self.nu0 = lib_data["nu0"]
                    self.amplitude = lib_data["amplitude"]
                    self.phase = lib_data["phase"]
                else:
                    raise KeyError
        self.recalculate()
        
    def compute(self,DS,EF,kwargs):
        #print("Computing Wavefield_NuT ...")
        if not DS.type == "intensity":
            raise TypeError
        if not EF.type == "electric field":
            raise TypeError
        self.fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        self.tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        self.nu0 = kwargs.get("nu0",DS.nu0)
        
        #- prepare data
        data = np.copy(DS.DS)
        data[data<0.] = 0.
        E = np.sqrt(data)*np.exp(1.j*EF.phase)
        
        self.WF = np.fft.fftshift(np.fft.fft2(E,axes=(0,1)),axes=(0,1))
        
        self.amplitude = np.abs(self.WF)
        self.phase = np.angle(self.WF)
        
class SkyMap:
    type = "sky map"
    def __init__(self,WF,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SkyMap.npz")
        if self.data_path==None:
            self.compute(DS,EF,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if WF==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(WF,kwargs)
                np.savez(file_data,th_par=self.th_par,th_ort=self.th_ort,SM=self.SM)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.th_par = lib_data["th_par"]
                    self.th_ort = lib_data["th_ort"]
                    self.SM = lib_data["SM"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_par,self.N_ort = self.SM.shape
    
    def compute(self,WF,kwargs):
        if not WF.type == "wavefield":
            raise TypeError
            
            
        self.nu0 = kwargs.get("nu0",1.4e+9)
        self.Deff = kwargs.get("Deff",120.*pc)
        self.fD_max = kwargs.get("fD_max",20.*mHz)
        self.fD_width = kwargs.get("fD_width",5.*mHz)
        self.N_par = kwargs.get("N_par",100)
        self.N_ort = kwargs.get("N_ort",100)
        self.veff_par = kwargs.get("veff_par",20.*kms)
        self.veff_ort = kwargs.get("veff_ort",20.*kms)
        
        #preparations        
        # - create containers
        fD_to_th_par = v_c/(self.nu0*self.veff_par)
        fD_to_th_ort = v_c/(self.nu0*self.veff_ort)
        f_tau = self.Deff/(2.*v_c)
        f_fD_par = -(self.nu0*self.veff_par)/v_c
        f_fD_ort = -(self.nu0*self.veff_ort)/v_c
        self.th_par = fD_to_th_par*np.linspace(-self.fD_max,self.fD_max,num=self.N_par,endpoint=True)
        self.th_ort = fD_to_th_ort*np.linspace(-self.fD_width/2.,self.fD_width/2.,num=self.N_ort,endpoint=True)
        self.SM = np.zeros((self.N_par,self.N_ort),dtype=float)
        # - get pixel boundaries of thetas
        dth_par = np.diff(self.th_par).mean()
        dth_ort = np.diff(self.th_ort).mean()
        lth_par = self.th_par - dth_par/2.
        rth_par = self.th_par + dth_par/2.
        lth_ort = self.th_ort - dth_ort/2.
        rth_ort = self.th_ort + dth_ort/2.
        # - get width of axes
        l_fD = WF.fD[-1]-WF.fD[0]
        l_tau = WF.tau[-1]-WF.tau[0]
        # - get pixel boundaries of SS
        lfD = WF.fD - WF.dfD/2.
        rfD = WF.fD + WF.dfD/2.
        ltau = WF.tau - WF.dtau/2.
        rtau = WF.tau + WF.dtau/2.
        
        #perform the computation
        # - main computation
        bar = progressbar.ProgressBar(maxval=self.N_par, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_par in range(self.N_par):
            bar.update(i_par)
            for i_ort in range(self.N_ort):
                # - compute bin boundaries in SS space
                v_fD_A = f_fD_par*lth_par[i_par] + f_fD_ort*lth_ort[i_ort]
                v_fD_B = f_fD_par*lth_par[i_par] + f_fD_ort*rth_ort[i_ort]
                v_fD_C = f_fD_par*rth_par[i_par] + f_fD_ort*lth_ort[i_ort]
                v_fD_D = f_fD_par*rth_par[i_par] + f_fD_ort*rth_ort[i_ort]
                v_tau_A = f_tau*(lth_par[i_par]**2+lth_ort[i_ort]**2)
                v_tau_B = f_tau*(lth_par[i_par]**2+rth_ort[i_ort]**2)
                v_tau_C = f_tau*(rth_par[i_par]**2+lth_ort[i_ort]**2)
                v_tau_D = f_tau*(rth_par[i_par]**2+rth_ort[i_ort]**2)
                v_fD_l = np.min([v_fD_A,v_fD_B,v_fD_C,v_fD_D])
                v_fD_u = np.max([v_fD_A,v_fD_B,v_fD_C,v_fD_D])
                v_tau_l = np.min([v_tau_A,v_tau_B,v_tau_C,v_tau_D])
                v_tau_u = np.max([v_tau_A,v_tau_B,v_tau_C,v_tau_D])
                area = np.abs((v_fD_u-v_fD_l)*(v_tau_u-v_tau_l))/(WF.dfD*WF.dtau)
                # - determine indices in SS space
                i_fD_l = int(np.rint((v_fD_l-WF.fD[0])/l_fD*(WF.N_fD-1)))
                i_fD_u = int(np.rint((v_fD_u-WF.fD[0])/l_fD*(WF.N_fD-1)))
                i_tau_l = int(np.rint((v_tau_l-WF.tau[0])/l_tau*(WF.N_tau-1)))
                i_tau_u = int(np.rint((v_tau_u-WF.tau[0])/l_tau*(WF.N_tau-1)))
                
                # i_fD_l1 = np.argmin(np.abs(v_fD_l-WF.fD))
                # i_fD_u1 = np.argmin(np.abs(v_fD_u-WF.fD))
                # i_tau_l1 = np.argmin(np.abs(v_tau_l-WF.tau))
                # i_tau_u1 = np.argmin(np.abs(v_tau_u-WF.tau))
                # if i_fD_l!=i_fD_l1:
                    # print(i_fD_l,i_fD_l1)
                # if i_fD_u!=i_fD_u1:
                    # print(i_fD_u,i_fD_u1)
                # if i_tau_l!=i_tau_l1:
                    # print(i_tau_l,i_tau_l1)
                # if i_tau_u!=i_tau_u1:
                    # print(i_tau_u,i_tau_u1)
                # - read in area weighted values
                for i_fD in range(i_fD_l,i_fD_u+1):
                    for i_tau in range(i_tau_l,i_tau_u+1):
                        if i_tau<WF.N_tau:
                            # - compute fractional area of pixel
                            frarea = (np.min([v_fD_u,rfD[i_fD]])-np.max([v_fD_l,lfD[i_fD]]))*(np.min([v_tau_u,rtau[i_tau]])-np.max([v_tau_l,ltau[i_tau]]))
                            self.SM[i_par,i_ort] += WF.amplitude[i_fD,i_tau]*frarea
                            # if frarea<0. or area<=0:
                                # print(frarea,area)
                                # print(v_fD_u,v_fD_l,v_tau_u,v_tau_l)
                                # print(rfD[i_fD],lfD[i_fD],rtau[i_tau],ltau[i_tau])
                                # raise ValueError
                self.SM[i_par,i_ort] /= area
        bar.finish()
        
    def crop(self,th_par_min=None,th_par_max=None,th_ort_min=None,th_ort_max=None,N_par=None,N_ort=None):
        # - create subset of data
        if th_par_min==None:
            i0_par = 0
        else:
            i0_par = np.argmin(np.abs(self.th_par-th_par_min))
        if th_par_max==None:
            i1_par = self.N_par
            if not N_par==None:
                if N_par <= i1_par:
                    i1_par = N_par
                else:
                    print("/!\ N_par too large! Using available data instead.")
        else:
            i1_par = np.argmin(np.abs(self.th_par-th_par_max))
            if not N_par==None and N_par!=i1_par:
                print("/!\ N_par incompatible with th_par_max! Using only th_par_max instead.")
        if th_ort_min==None:
            i0_ort = 0
        else:
            i0_ort = np.argmin(np.abs(self.th_ort-th_ort_min))
        if th_ort_max==None:
            i1_ort = self.N_ort
            if not N_ort==None:
                if N_ort <= i1_ort:
                    i1_ort = N_ort
                else:
                    print("/!\ N_nu too large! Using available data instead.")
        else:
            i1_ort = np.argmin(np.abs(self.th_ort-th_ort_max))
            if not N_ort==None and N_ort!=i1_ort:
                print("/!\ N_nu incompatible with th_ort_max! Using only th_ort_max instead.")
        if i0_par!=0 or i1_par!=self.N_par or i0_ort!=0 or i1_ort!=self.N_ort:
            self.th_par = self.th_par[i0_par:i1_par]
            self.th_ort = self.th_ort[i0_ort:i1_ort]
            self.SM = self.SM[i0_par:i1_par,i0_ort:i1_ort]
            self.recalculate()
        
class FeatureAlignment:
    type = "feature alignment"
    def __init__(self,SM,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        if self.data_path==None:
            self.compute(SM,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,"FeatureAlignment.npz")
            recompute = True
            if SM==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(SM,kwargs)
                np.savez(file_data,th_par=self.th_par,th_ort=self.th_ort,FA=self.FA,th1=self.th1,th2=self.th2)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.th_par = lib_data["th_par"]
                    self.th_ort = lib_data["th_ort"]
                    self.th1 = lib_data["th1"]
                    self.th2 = lib_data["th2"]
                    self.FA = lib_data["FA"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_par,self.N_ort = self.FA.shape
        self.dth_par = np.diff(self.th_par).mean()
        self.dth_ort = np.diff(self.th_ort).mean()
        self.th1_min = np.min(self.th1)
        self.th1_max = np.max(self.th1)
        self.th2_min = np.min(self.th2)
        self.th2_max = np.max(self.th2)
        
    def compute(self,SM,kwargs):
        if not SM.type == "sky map":
            raise TypeError
            
        self.N_par = SM.N_par
        self.N_ort = SM.N_ort
        self.th_par = np.copy(SM.th_par)
        self.th_ort = np.copy(SM.th_ort)
        self.th1 = np.copy(self.th_par)
        self.th2 = np.copy(self.th_ort)
        self.FA = np.copy(SM.SM)
        
    def addSkyMap(self,SM,**kwargs):
        theta_los = kwargs.get("theta_los",0.*mas)
        file_data = os.path.join(self.data_path,"FeatureAlignment.npz")
        
        #regrid the new Map to the old one
        new_N_par = int((SM.th_par[-1]-SM.th_par[0])/self.dth_par)+1
        new_N_ort = int((SM.th_ort[-1]-SM.th_ort[0])/self.dth_ort)+1
        new_th_par = np.linspace(0,new_N_par-1,num=new_N_par,endpoint=True,dtype=float)*self.dth_par+SM.th_par[0]
        new_th_ort = np.linspace(0,new_N_ort-1,num=new_N_ort,endpoint=True,dtype=float)*self.dth_ort+SM.th_ort[0]
        xg, yg = np.meshgrid(new_th_par, new_th_ort,indexing='ij')
        points = np.stack((xg, yg),axis=-1)
        
        ip = interp.RegularGridInterpolator((SM.th_par,SM.th_ort),SM.SM,method='nearest')
        new_data = (ip(points)).reshape(new_N_par,new_N_ort)
        
        shift = int(np.rint(theta_los/self.dth_par))
        new_th_par += shift*self.dth_par
        #new_th_ort += self.th2_max+self.dth_ort
        
        dmin = self.th1_min-new_th_par[0]
        N_min = int(np.rint(np.abs(dmin)/self.dth_par))
        if dmin>0.:
            self.FA = np.concatenate((np.zeros((N_min,self.N_ort)),self.FA),axis=0)
            self.th1_min = new_th_par[0]
        elif dmin<0.:
            new_data = np.concatenate((np.zeros((N_min,new_N_ort)),new_data),axis=0)
            
        dmax = self.th1_max-new_th_par[-1]
        N_max = int(np.rint(np.abs(dmax)/self.dth_par))
        if dmax<0.:
            self.FA = np.concatenate((self.FA,np.zeros((N_max,self.N_ort))),axis=0)
            self.th1_max = new_th_par[-1]
        elif dmax>0.:
            new_data = np.concatenate((new_data,np.zeros((N_max,new_N_ort))),axis=0)
            
        new_N_par,new_N_ort = new_data.shape
        if new_N_par<self.FA.shape[0]:
            new_data = np.concatenate((new_data,np.zeros((self.FA.shape[0]-new_N_par,new_N_ort))),axis=0)
        elif new_N_par-self.FA.shape[0]==1:
            new_data = np.delete(new_data,-1,axis=0)
        elif new_N_par-self.FA.shape[0]==2:
            new_data = np.delete(new_data,(-2,-1),axis=0)
            
        self.FA = np.concatenate((self.FA,new_data),axis=1)
        self.N_par,self.N_ort = self.FA.shape
        self.th1 = np.linspace(self.th1_min,self.th1_max,num=self.N_par)
        self.th2 = np.arange(self.N_ort)*self.dth_ort
        
        np.savez(file_data,th_par=self.th_par,th_ort=self.th_ort,FA=self.FA,th1=self.th1,th2=self.th2)
        
class SecDS:
    type = "secondary dynamic spectrum"
    def __init__(self,eigens,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","SecDS.npz")
        if self.data_path==None:
            self.compute(eigens,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if eigens==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(eigens,kwargs)
                np.savez(file_data,t=self.t,nu=self.nu,DS=self.DS)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.t = lib_data["t"]
                    self.nu = lib_data["nu"]
                    self.DS = lib_data["DS"]
                else:
                    raise KeyError
        self.recalculate()
        
    def slice(self,i0_t=0,i1_t=-1,i0_nu=0,i1_nu=-1):
        self.t = self.t[i0_t:i1_t]
        self.nu = self.nu[i0_nu:i1_nu]
        self.DS = self.DS[i0_t:i1_t,i0_nu:i1_nu]
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.DS.shape
        self.dt = self.t[1]-self.t[0]
        self.dnu = self.nu[1]-self.nu[0]
        self.t_min = self.t[0]
        self.t_max = self.t[-1]
        self.nu_min = self.nu[0]
        self.nu_max = self.nu[-1]
        self.timespan = self.t_max-self.t_min
        self.bandwidth = self.nu_max-self.nu_min
        self.t0 = np.mean(self.t)
        self.nu0 = np.mean(self.nu)
        
    def compute(self,eigens,kwargs):
        if eigens.type == "eigenvectors":
            slope = kwargs.get("slope",1.)
                
            self.DS = np.zeros((eigens.N_th,eigens.N_nu),dtype=float)
            weights = np.zeros((eigens.N_th),dtype=int)
            
            mu = np.mean(np.abs(eigens.eigenvectors),axis=1)
            data = np.abs(eigens.eigenvectors)/mu[:,na,:]
            self.t = -eigens.th/slope
            self.nu = np.copy(eigens.nu)
                
            bar = progressbar.ProgressBar(maxval=eigens.N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i_t in range(eigens.N_t):
                bar.update(i_t)
                for i_th in range(eigens.N_th):
                    ttilde = eigens.th[i_th]-slope*(eigens.t[i_t]-eigens.t0)
                    i_tt = np.argmin(np.abs(eigens.th-ttilde))
                    self.DS[i_tt,:] += data[i_th,i_t,:]
                    weights[i_tt] += 1
            bar.finish()
            weights[weights==0] = 1
            self.DS = self.DS/weights[:,na]
            self.DS = np.flip(self.DS,axis=0)
            self.t = np.flip(self.t) - np.min(self.t)
        elif eigens.type == "brightness distribution":
            vm = kwargs.get("vm",1.)
                
            self.DS = np.zeros((eigens.N_stau,eigens.N_nu),dtype=float)
            weights = np.zeros((eigens.N_stau),dtype=int)
            
            b = np.mean(np.abs(eigens.mu),axis=1)
            data = np.abs(eigens.mu)/b[:,na,:]
            self.t = -eigens.stau/vm
            self.nu = np.copy(eigens.nu)
                
            bar = progressbar.ProgressBar(maxval=eigens.N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for i_t in range(eigens.N_t):
                bar.update(i_t)
                for i_stau in range(eigens.N_stau):
                    ttilde = eigens.stau[i_stau]-vm*(eigens.t[i_t]-eigens.t0)
                    i_tt = np.argmin(np.abs(eigens.stau-ttilde))
                    self.DS[i_tt,:] += data[i_stau,i_t,:]
                    weights[i_tt] += 1
            bar.finish()
            weights[weights==0] = 1
            self.DS = self.DS/weights[:,na]
            self.DS = np.flip(self.DS,axis=0)
            self.t = np.flip(self.t) - np.min(self.t)
        else:
            raise TypeError
        
class ACF_DS:
    type = "autocorrelation"
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","ACF_DS.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,t_shift=self.t_shift,nu_shift=self.nu_shift,ACF=self.ACF)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.t_shift = lib_data["t_shift"]
                    self.nu_shift = lib_data["nu_shift"]
                    self.ACF = lib_data["ACF"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_nu = self.ACF.shape
        
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        compute_direct = kwargs.get("compute_direct",False)
        if compute_direct:
            t_sampling = kwargs.get("t_sampling",1)
            nu_sampling = kwargs.get("nu_sampling",1)
            
            # - downsampling
            data = block_reduce(DS.DS, block_size=(t_sampling,nu_sampling), func=np.mean)
            coordinates = np.array([DS.t,DS.t])
            coordinates = block_reduce(coordinates, block_size=(1,t_sampling), func=np.mean, cval=DS.t[-1])
            self.t_shift = coordinates[0,:]
            coordinates = np.array([DS.nu,DS.nu])
            coordinates = block_reduce(coordinates, block_size=(1,nu_sampling), func=np.mean, cval=DS.nu[-1])
            self.nu_shift = coordinates[0,:]
            
            self.t_shift = self.t_shift - np.mean(self.t_shift)
            self.nu_shift = self.nu_shift - np.mean(self.nu_shift)
            data = (data - np.mean(data))/np.std(data)
            
            self.ACF = scipy.signal.correlate2d(data,data,mode='same')
        else:
            self.ACF = np.fft.fftshift(np.fft.ifft2( np.fft.fft2(DS.DS) * np.fft.fft2(DS.DS).conj() ))
            self.ACF = np.real(self.ACF)
            self.t_shift = np.linspace(-DS.timespan/2., DS.timespan/2., DS.N_t, endpoint=DS.N_t%2)
            self.nu_shift = np.linspace(-DS.bandwidth/2., DS.bandwidth/2., DS.N_nu, endpoint=DS.N_nu%2)
            
    def fit_tscale(self,**kwargs):
        # Take slice through center of cross-correlation along time axis
        # Ignoring zero component with noise-noise correlation
        ACF_shifted = np.fft.ifftshift(self.ACF)
        ccorr_t = ACF_shifted[:,1] + ACF_shifted[:,-1]
        ccorr_t -= np.median(ccorr_t)
        ccorr_t /= np.max(ccorr_t)
        ccorr_t_shift = np.copy(ccorr_t)
        ccorr_t = np.fft.fftshift(ccorr_t)
        
        def Gaussian(x, sigma, A, C):
            return A*np.exp( -x**2 / (2*sigma**2) ) + C
            
        # Fit the slices in frequency and time with a Gaussian
        # p0 values are just a starting guess
        dt = self.t_shift[1] - self.t_shift[0]
        is_sigma = np.argwhere(ccorr_t_shift<np.exp(-1./2.))
        i_sigma = np.min(is_sigma)
        p0 = [i_sigma*dt, 1., 0]
        popt, pcov = curve_fit(Gaussian, self.t_shift, ccorr_t, p0=p0)
        #tscint = np.sqrt(2) * abs(popt[0])
        #tscinterr = np.sqrt(2) * np.sqrt(pcov[0,0])
        tscint = abs(popt[0])
        tscinterr = np.sqrt(pcov[0,0])
        t_model_ACF = Gaussian(self.t_shift, popt[0], popt[1], popt[2])
        
        return tscint,tscinterr,ccorr_t,t_model_ACF
        
    def fit_nuscale(self,**kwargs):
        # Take slice through center of cross-correlation along freq axis
        # Ignoring zero component with noise-noise correlation
        ACF_shifted = np.fft.ifftshift(self.ACF)
        ccorr_nu = ACF_shifted[1] + ACF_shifted[-1]
        ccorr_nu -= np.median(ccorr_nu)
        ccorr_nu /= np.max(ccorr_nu)
        ccorr_nu_shift = np.copy(ccorr_nu)
        ccorr_nu = np.fft.fftshift(ccorr_nu)
        
        def Gaussian(x, sigma, A, C):
            return A*np.exp( -x**2 / (2*sigma**2) *np.log(2) ) + C
        
        def Lorentzian(x, sigma, A, C):
            return A/( 1 + x**2 / (2*sigma**2) ) + C
            
        # Fit the slices in frequency and time with a Gaussian
        # p0 values are just a starting guess
        dnu_fit_max = kwargs.get("dnu_fit_max",np.max(np.abs(self.nu_shift)))
        dnu = self.nu_shift[1] - self.nu_shift[0]
        is_sigma = np.argwhere(ccorr_nu_shift<np.exp(-1./2.))
        i_sigma = np.min(is_sigma)
        nuiss_0 = kwargs.get("muiss_0",i_sigma*dnu)
        p0 = [nuiss_0, 1., 0]
        popt, pcov = curve_fit(Lorentzian, self.nu_shift[np.abs(self.nu_shift)<=dnu_fit_max], ccorr_nu[np.abs(self.nu_shift)<=dnu_fit_max], p0=p0)
        #nuscint = np.sqrt(2*np.log(2)) * abs(popt[0])
        #nuscinterr = np.sqrt(2*np.log(2)) * np.sqrt(pcov[0,0])
        nuscint = abs(popt[0])
        nuscinterr = np.sqrt(pcov[0,0])
        nu_model_ACF = Lorentzian(self.nu_shift, popt[0], popt[1], popt[2])
        
        return nuscint,nuscinterr,ccorr_nu,nu_model_ACF
        
        
class EV_backtrafo(eigenvectors):
        
    def compute(self,DS,kwargs):
        #load and check data
        if not DS.type == "intensity":
            raise TypeError
            
        N_th = kwargs.get("N_th",100)
        fD_max = kwargs.get("fD_max",1.*mHz)
        tchunk = kwargs.get("tchunk",50)
        nuchunk = kwargs.get("nuchunk",300)
        Deff = kwargs.get("Deff",120.*pc)
        veff = kwargs.get("veff",20.*kms)
        fD_width_inner = kwargs.get("fD_width_inner",1.)
        fD_width_outer = kwargs.get("fD_width_outer",2.)
        tau_width_inner = kwargs.get("tau_width_inner",1.)
        tau_width_outer = kwargs.get("tau_width_outer",2.)
        stripe = kwargs.get("stripe",False)
        
        #preparations
        # - Fourier transformation
        fD = np.fft.fftshift(np.fft.fftfreq(DS.N_t,DS.dt))
        tau = np.fft.fftshift(np.fft.fftfreq(DS.N_nu,DS.dnu))
        CS = np.fft.fftshift(np.fft.fft2(DS.DS,axes=(0,1)),axes=(0,1))
        N_fD,N_tau = CS.shape
        dfD = np.diff(fD).mean()
        dtau = np.diff(tau).mean()
        # - downsampling
        coordinates = np.array([DS.t,DS.t])
        coordinates = block_reduce(coordinates, block_size=(1,tchunk), func=np.mean, cval=DS.t[-1])
        self.t = coordinates[0,:]
        coordinates = np.array([DS.nu,DS.nu])
        coordinates = block_reduce(coordinates, block_size=(1,nuchunk), func=np.mean, cval=DS.nu[-1])
        self.nu = coordinates[0,:]
        self.N_t = len(self.t)
        self.N_nu = len(self.nu)
        # - theta scaling
        fD_to_rad = v_c/DS.nu0/veff
        rad_to_fD = 1./fD_to_rad
        self.th = np.linspace(-fD_max*fD_to_rad,fD_max*fD_to_rad,num=N_th,dtype=float,endpoint=True)
        self.eigenvectors = np.empty((N_th,self.N_t,self.N_nu),dtype=float)
        eta = v_c*Deff/(2.*DS.nu0**2*veff**2)
        # - define mask
        fD_freq = np.pi/2./(fD_width_outer-fD_width_inner)
        tau_freq = np.pi/2./(tau_width_outer-tau_width_inner)
        def mask(dfD,dtau):
            if dfD>fD_width_outer:
                factor = 0.
            elif dfD>fD_width_inner:
                factor = np.cos(fD_freq*(dfD-fD_width_inner))**2
            else:
                factor = 1.
            if dtau>tau_width_outer:
                factor *= 0.
            elif dtau>tau_width_inner:
                factor *= np.cos(tau_freq*(dtau-tau_width_inner))**2
            return factor
        arr = np.zeros((N_fD,N_tau))
        for i_fD,v_fD in enumerate(fD):
            if np.abs(v_fD)<fD_width_outer:
                for i_tau,v_tau in enumerate(tau):
                    if np.abs(v_tau)<tau_width_outer:
                        arr[i_fD,i_tau] = mask(np.abs(v_fD),np.abs(v_tau))
        if not stripe:
            for i_fD,v_fD in enumerate(fD):
                col = arr[i_fD,:]
                shift = int(np.rint(eta*v_fD**2/dtau))
                col = np.roll(col,shift)
                arr[i_fD,:] = col
        
        #main computation
        bar = progressbar.ProgressBar(maxval=N_th, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in range(N_th):
            bar.update(i_th)
            fD0 = self.th[i_th]*rad_to_fD
            tau0 = eta*fD0**2
            di_fD = int(np.rint(fD0/dfD))
            di_tau = int(np.rint(tau0/dtau))
            f_mask = np.roll(arr,(di_fD,di_tau),axis=(0,1)) + np.roll(arr,(-di_fD,-di_tau),axis=(0,1))
            f_mask[f_mask>1.] = 1.
            source = np.fft.ifftshift(CS*f_mask,axes=(0,1))
            # - Fourier backtransformation
            result = np.abs(np.fft.ifft2(source,axes=(0,1)))
            self.eigenvectors[i_th,:,:] = block_reduce(result, block_size=(tchunk,nuchunk), func=np.mean)
        bar.finish()
            
class zeta_eigenvalues:
    type = "zeta eigenvalues"
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","zeta_eigenvalues.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,zetas=self.zetas,EV=self.EV,EVs=self.EVs,t=self.t,mjd=self.mjd)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.zetas = lib_data["zetas"]
                    self.EV = lib_data["EV"]
                    try:
                        self.t = lib_data["t"]
                        self.mjd = lib_data["mjd"]
                        self.EVs = lib_data["EVs"]
                    except:
                        self.t = np.zeros(1)
                        self.mjd = np.zeros_like(self.t)
                        self.EVs = np.zeros((len(self.t),len(self.zetas)))
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_zeta = len(self.zetas)
        
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        N_th = kwargs.get("N_th",51)
        tchunk = kwargs.get("tchunk",60)
        nuchunk = kwargs.get("nuchunk",200)
        npad = kwargs.get("npad",3)
        tau_max = kwargs.get("tau_max",15.*musec)
        N_zeta = kwargs.get("N_zeta",100)
        zeta_min = kwargs.get("zeta_min",0.6e-9)
        zeta_max = kwargs.get("zeta_max",2.2e-9)
        vary_chunk = kwargs.get("vary_chunk",True)
        
        stau_max = np.sqrt(tau_max)
        
        if vary_chunk:
            def find_chunks(N,Nc):
                """
                N : length of list
                Nc : Maximum number of entries per chunk
                """
                #number of chunks
                Ns = (2*N)//Nc
                #optimal shift of chunk (float)
                shift = N/(Ns+1.)
                #create list by rounding to integers
                starts = [np.rint(i*shift).astype(int) for i in range(Ns)]
                #mids = [np.rint((i+1)*shift).astype(int) for i in range(Ns)]
                ends = [np.rint((i+2)*shift).astype(int) for i in range(Ns)]
                return starts,ends,int(shift)
            
            #- split into overlapping chunks
            t_starts,t_ends,t_shift = find_chunks(DS.N_t,tchunk)
            nu_starts,nu_ends,nu_shift = find_chunks(DS.N_nu,nuchunk)
            N_tchunk = len(t_starts)
            N_nuchunk = len(nu_starts)
            
            staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
            self.EV = np.zeros(N_zeta)
            self.EVs = np.zeros((N_tchunk,N_zeta))
            self.mjd = np.zeros(N_tchunk)
            self.t = np.zeros(N_tchunk)
            self.zetas = np.linspace(zeta_min,zeta_max,num=N_zeta)
            
            #main computation
            for tc in range(N_tchunk):
                t_chunk = DS.t[t_starts[tc]:t_ends[tc]]
                mjd_chunk = DS.mjd[t_starts[tc]:t_ends[tc]]
                self.t[tc] = np.mean(t_chunk)
                self.mjd[tc] = np.mean(mjd_chunk)
                print("{0}/{1}: {2} - {3} ({4})".format(tc+1,N_tchunk,t_chunk[0],t_chunk[-1],len(t_chunk)))
                bar = progressbar.ProgressBar(maxval=N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for fc in range(N_nuchunk):
                    bar.update(fc)
                    #print(tc,fc)
                    dspec = DS.DS[t_starts[tc]:t_ends[tc],nu_starts[fc]:nu_ends[fc]]
                    dspec = dspec - dspec.mean()
                    nu_chunk = DS.nu[nu_starts[fc]:nu_ends[fc]]
                    f0_evo = nu_chunk.mean()
                    dspec_pad = np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())

                    ##Form SS and coordinate arrays
                    SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                    fd = np.fft.fftshift(np.fft.fftfreq((npad+1)*t_chunk.shape[0],t_chunk[1]-t_chunk[0]))
                    tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nu_chunk.shape[0],nu_chunk[1]-nu_chunk[0]))
                    
                    ##Setup for chisq search
                    eigs_zeta = np.zeros(N_zeta)
                    
                    ##Determine chisq for each delay drift
                    for i in range(N_zeta):
                        zeta = self.zetas[i]
                        stau_to_fD = 2.*f0_evo*zeta
                        th1 = np.ones((N_th,N_th))*staus
                        th2 = th1.T
                        dfd = np.diff(fd).mean()
                        dtau = np.diff(tau).mean()
                        tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                        fd_inv = ((stau_to_fD*(th1-th2)-fd[0]+dfd/2)//dfd).astype(int)
                        thth = np.zeros((N_th,N_th), dtype=complex)
                        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv > 0) * (fd_inv < fd.shape[0])
                        thth[pnts] = SS[fd_inv[pnts],tau_inv[pnts]]
                        eta = 1./(2.*f0_evo*zeta)**2
                        thth *= np.sqrt(np.abs(2*eta*stau_to_fD*(th2-th1))) #flux conervation
                        #thth *= np.abs(2*eta*stau_to_fD*(th2-th1)) #Jacobian
                        thth /= np.mean(np.abs(thth))
                        if 1:
                            thth -= np.tril(thth) #make hermitian
                            thth += np.conjugate(np.triu(thth).T)
                            thth -= np.diag(np.diag(thth))
                            thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                            thth = np.nan_to_num(thth)
                        else:
                            #Produces a similar but slightly different result
                            thth = (thth+np.conj(np.transpose(thth)))/2. #assert hermitian
                        ##Find first eigenvector and value
                        v0 = thth[thth.shape[0]//2,:]
                        v0 /= np.sqrt((np.abs(v0)**2).sum())
                        try:
                            w,V = eigsh(thth,1,v0=v0,which='LA')
                            eigs_zeta[i] = np.abs(w[0])
                        except:
                            print("did not find any eigenvalues to sufficient accuracy")
                            eigs_zeta[i] = np.nan
                        
                    if not np.isnan(eigs_zeta).any():
                        self.EV += eigs_zeta
                        self.EVs[tc,:] += eigs_zeta
                bar.finish()
        else:
            hnuchunk = int(nuchunk/2)
            htchunk = int(tchunk/2)
            N_tchunk = (DS.N_t-htchunk)//htchunk
            N_nuchunk = (DS.N_nu-hnuchunk)//hnuchunk
            staus = np.linspace(-stau_max,stau_max,num=N_th,dtype=float,endpoint=True)
            self.EV = np.zeros(N_zeta)
            self.EVs = np.zeros((N_tchunk,N_zeta))
            self.mjd = np.zeros(N_tchunk)
            self.t = np.zeros(N_tchunk)
            self.zetas = np.linspace(zeta_min,zeta_max,num=N_zeta)
            
            #main computation
            bar = progressbar.ProgressBar(maxval=N_tchunk*N_nuchunk, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            for tc in range(N_tchunk):
                t_chunk = DS.t[tc*htchunk:tc*htchunk+tchunk]
                mjd_chunk = DS.mjd[tc*htchunk:tc*htchunk+tchunk]
                self.t[tc] = np.mean(t_chunk)
                self.mjd[tc] = np.mean(mjd_chunk)
                for fc in range(N_nuchunk):
                    bar.update(tc*N_nuchunk+fc)
                    #print(tc,fc)
                    dspec = DS.DS[tc*htchunk:tc*htchunk+tchunk,fc*hnuchunk:fc*hnuchunk+nuchunk]
                    dspec = dspec - dspec.mean()
                    nu_chunk = DS.nu[fc*hnuchunk:fc*hnuchunk+nuchunk]
                    f0_evo = nu_chunk.mean()
                    dspec_pad = np.pad(dspec,((0,npad*dspec.shape[0]),(0,npad*dspec.shape[1])),mode='constant',constant_values=dspec.mean())

                    ##Form SS and coordinate arrays
                    SS = np.fft.fftshift(np.fft.fft2(dspec_pad))
                    fd = np.fft.fftshift(np.fft.fftfreq((npad+1)*t_chunk.shape[0],t_chunk[1]-t_chunk[0]))
                    tau = np.fft.fftshift(np.fft.fftfreq((npad+1)*nu_chunk.shape[0],nu_chunk[1]-nu_chunk[0]))
                    
                    ##Setup for chisq search
                    eigs_zeta = np.zeros(N_zeta)
                    
                    ##Determine chisq for each delay drift
                    for i in range(N_zeta):
                        zeta = self.zetas[i]
                        stau_to_fD = 2.*f0_evo*zeta
                        th1 = np.ones((N_th,N_th))*staus
                        th2 = th1.T
                        dfd = np.diff(fd).mean()
                        dtau = np.diff(tau).mean()
                        tau_inv = (((th1**2-th2**2)-tau[0]+dtau/2)//dtau).astype(int)
                        fd_inv = ((stau_to_fD*(th1-th2)-fd[0]+dfd/2)//dfd).astype(int)
                        thth = np.zeros((N_th,N_th), dtype=complex)
                        pnts = (tau_inv > 0) * (tau_inv < tau.shape[0]) * (fd_inv > 0) * (fd_inv < fd.shape[0])
                        thth[pnts] = SS[fd_inv[pnts],tau_inv[pnts]]
                        eta = 1./(2.*f0_evo*zeta)**2
                        thth *= np.sqrt(np.abs(2*eta*stau_to_fD*(th2-th1))) #flux conervation
                        thth /= np.mean(np.abs(thth))
                        if 1:
                            thth -= np.tril(thth) #make hermitian
                            thth += np.conjugate(np.triu(thth).T)
                            thth -= np.diag(np.diag(thth))
                            thth -= np.diag(np.diag(thth[::-1, :]))[::-1, :]
                            thth = np.nan_to_num(thth)
                        else:
                            #Produces a similar but slightly different result
                            thth = (thth+np.conj(np.transpose(thth)))/2. #assert hermitian
                        ##Find first eigenvector and value
                        v0 = thth[thth.shape[0]//2,:]
                        v0 /= np.sqrt((np.abs(v0)**2).sum())
                        w,V = eigsh(thth,1,v0=v0,which='LA')
                        eigs_zeta[i] = np.abs(w[0])
                        
                    if not np.isnan(eigs_zeta).any():
                        self.EV += eigs_zeta
                        self.EVs[tc,:] += eigs_zeta
            bar.finish()
        
    def fit_peak(self,zeta_range):
        def chi_par(x, A, x0, C):
            """
            Parabola for fitting to chisq curve.
            """
            return A*(x - x0)**2 + C
        z_max = self.zetas[self.EV==self.EV.max()][0]
        zetas_fit = self.zetas[np.abs(self.zetas-z_max)<zeta_range/2.]
        eigs_fit = self.EV[np.abs(self.zetas-z_max)<zeta_range/2.]
        C = eigs_fit.max()
        x0 = zetas_fit[eigs_fit==C][0]
        A = (eigs_fit[0]-C)/((zetas_fit[0]-x0)**2)
        popt,pcov = curve_fit(chi_par,zetas_fit,eigs_fit,p0=np.array([A,x0,C]))
        zeta_fit = popt[1]
        zeta_sig = np.sqrt(-(eigs_fit-chi_par(zetas_fit,*popt)).std()/popt[0])
        fit_result = chi_par(zetas_fit, *popt)
        
        return zeta_fit,zeta_sig,zetas_fit,fit_result
        
    def fit_double_peak(self,zeta_range):
        def chi_par(x, A, xA, sA, C):
            """
            Fitting two gaussian components.
            """
            return A*np.exp(-0.5*(x - xA)**2/sA**2) + C
        z_max = self.zetas[self.EV==self.EV.max()][0]
        zetas_fit = self.zetas[np.abs(self.zetas-z_max)<zeta_range/2.]
        eigs_fit = self.EV[np.abs(self.zetas-z_max)<zeta_range/2.]
        C = eigs_fit.max()
        xA = zetas_fit[eigs_fit==C][0]
        A = C-eigs_fit[0]
        sA = zeta_range/2.
        popt1,pcov = curve_fit(chi_par,zetas_fit,eigs_fit,p0=np.array([A, xA, sA, C]),maxfev = 10000)
        perr1 = np.sqrt(np.diag(pcov))
        zeta1_fit = popt1[1]
        zeta1_sig = perr1[1]
        
        new_eig = self.EV-chi_par(self.zetas, popt1[0], popt1[1], popt1[2], 0.)
        z_max = self.zetas[new_eig==new_eig.max()][0]
        if z_max == new_eig[0]:
            new_eig[:int(0.1*len(new_eig))] = 0.
            z_max = self.zetas[new_eig==new_eig.max()][0]
        zetas_fit = self.zetas[np.abs(self.zetas-z_max)<zeta_range/2.]
        eigs_fit = new_eig[np.abs(self.zetas-z_max)<zeta_range/2.]
        
        try:
            C = eigs_fit.max()
            xA = zetas_fit[eigs_fit==C][0]
            A = C-eigs_fit[0]
            sA = zeta_range/2.
            popt2,pcov = curve_fit(chi_par,zetas_fit,eigs_fit,p0=np.array([A, xA, sA, C]),maxfev = 10000)
            perr2 = np.sqrt(np.diag(pcov))
            zeta2_fit = popt2[1]
            zeta2_sig = perr2[1]
        except:
            zeta2_fit = np.nan
            zeta2_sig = np.nan
            popt2 =[0.,1.,1.]
        
        fit_result = chi_par(self.zetas, popt1[0], popt1[1], popt1[2], 0.) + chi_par(self.zetas, popt2[0], popt2[1], popt2[2], 0.)
        fit_result += np.mean(self.EV-fit_result)
        
        return zeta1_fit,zeta1_sig,zeta2_fit,zeta2_sig,self.zetas,fit_result
           
class refracted_scintles:
    type = "refracted_scintles"
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","refracted_scintles.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,t=self.t,DMp=self.DMp,Hough=self.Hough)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.t = lib_data["t"]
                    self.DMp = lib_data["DMp"]
                    self.Hough = lib_data["Hough"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_DMp = self.Hough.shape
        
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        shift_max = kwargs.get("shift_max",1.*minute)
        N_DMp = kwargs.get("N_DMp",11)
        
        self.t = np.copy(DS.t)
        N_t = DS.N_t
        N_nu = DS.N_nu
        dt = DS.dt
        Dt = DS.timespan
        DM_slope_max = shift_max/(1./DS.nu[0]**2-1./DS.nu[-1]**2)
        self.DMp = np.linspace(-DM_slope_max,DM_slope_max,num=N_DMp,endpoint=True)
        self.Hough = np.empty((N_t,N_DMp),dtype=float)
        is_nu = np.arange(N_nu)
        
        bar = progressbar.ProgressBar(maxval=N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_t0,t0 in enumerate(self.t):
            bar.update(i_t0)
            for i_DMp,v_DMp in enumerate(self.DMp):
                #t_obs = t0 + f_refr*v_DMp/DS.nu**2
                t_obs = t0 + v_DMp/DS.nu**2
                is_t = (np.rint((t_obs-DS.t[0])/dt)).astype(int)
                is_t[is_t<0] = 0
                is_t[is_t>=N_t] = N_t-1
                DS_slice = DS.DS[is_t,is_nu]
                #print(np.std(DS_slice))
                
                self.Hough[i_t0,i_DMp] = np.sum(DS_slice)
        bar.finish()
        
    def get_peaks(self,**kwargs):
        std_window = kwargs.get("std_window",11)
        
        N2 = std_window
        N_std = int(2*N2)
        Hough_std = np.empty((self.N_t-N_std,self.N_DMp),dtype=float)
        t_std = self.t[N2:self.N_t-N2]
        N_t_std = len(t_std)
        bar = progressbar.ProgressBar(maxval=self.N_t-N_std, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_t0 in range(N2,self.N_t-N2):
            bar.update(i_t0-N2)
            Hough_std[i_t0-N2,:] = np.std(self.Hough[i_t0-N2:i_t0+N2,:],axis=0)
        bar.finish()
    
        is_peak = np.argmax(Hough_std,axis=1)
        DMp_peak = self.DMp[is_peak]
        peak_values = self.Hough[np.arange(N_t_std),is_peak]
        return DMp_peak,peak_values,Hough_std,t_std
        
class refracted_scintles_std:
    type = "refracted_scintles_std"
    def __init__(self,DS,**kwargs):
        self.data_path = kwargs.get("data_path",None)
        overwrite = kwargs.get("overwrite",True)
        file_name = kwargs.get("file_name","refracted_scintles_std.npz")
        if self.data_path==None:
            self.compute(DS,kwargs)
        else:
            if not os.path.exists(self.data_path):
                os.makedirs(self.data_path)
            file_data = os.path.join(self.data_path,file_name)
            recompute = True
            if DS==None:
                recompute = False
            elif not overwrite:
                if os.path.exists(file_data):
                    recompute = False
            if recompute:
                self.compute(DS,kwargs)
                np.savez(file_data,t=self.t,DMp=self.DMp,std_measure=self.std_measure)
            else:
                if os.path.exists(file_data):
                    lib_data = np.load(file_data)
                    self.t = lib_data["t"]
                    self.DMp = lib_data["DMp"]
                    self.std_measure = lib_data["std_measure"]
                else:
                    raise KeyError
        self.recalculate()
        
    def recalculate(self):
        #provide some useful parameters
        self.N_t,self.N_DMp = self.std_measure.shape
        
    def compute(self,DS,kwargs):
        if not DS.type == "intensity":
            raise TypeError
        shift_max = kwargs.get("shift_max",1.*minute)
        N_DMp = kwargs.get("N_DMp",11)
        
        self.t = np.copy(DS.t)
        N_t = DS.N_t
        N_nu = DS.N_nu
        dt = DS.dt
        DM_slope_max = shift_max/(1./DS.nu[0]**2-1./DS.nu[-1]**2)
        self.DMp = np.linspace(-DM_slope_max,DM_slope_max,num=N_DMp,endpoint=True)
        self.std_measure = np.empty((N_t,N_DMp),dtype=float)
        is_nu = np.arange(N_nu)
        
        bar = progressbar.ProgressBar(maxval=N_t, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_t0,t0 in enumerate(self.t):
            bar.update(i_t0)
            for i_DMp,v_DMp in enumerate(self.DMp):
                #t_obs = t0 + f_refr*v_DMp/DS.nu**2
                t_obs = t0 + v_DMp/DS.nu**2
                is_t = (np.rint((t_obs-DS.t[0])/dt)).astype(int)
                mask = np.ones(len(is_nu),dtype=bool)
                #is_t[is_t<0] = 0
                #is_t[is_t>=N_t] = N_t-1
                #DS_slice = DS.DS[is_t,is_nu]
                mask[is_t<0] = False
                mask[is_t>=N_t] = False
                DS_slice = DS.DS[is_t[mask],is_nu[mask]]
                #print(np.std(DS_slice))
                
                self.std_measure[i_t0,i_DMp] = np.std(DS_slice)
        bar.finish()