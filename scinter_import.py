import numpy as np
from numpy import newaxis as na
from astropy.io import fits
import os

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

import scinter_data

class B1508p55_Eff_classic(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508_Eff_2020/"
        if type(date)==str:
            if date in ["20200130","20200131","20200418","20200516","20200517","20200519","20200527","20200531","20200608"]:
                data_path = file_path+'dynspec_Rob_'+date+'.npz'
                data_npz = np.load(data_path)
                self.DS = data_npz["dynspec"]
                self.nu = data_npz["f_MHz"]*MHz
                N_t,N_nu = self.DS.shape
                t0 = float(data_npz['t0'])
                self.t = np.linspace(0.,(N_t-1)*10.,num=N_t,dtype=float,endpoint=True)
                self.mjd = t0+self.t/(24.*3600.)
            else:
                data_path = file_path+'dynspec_'+date+'.npz'
                data_npz = np.load(data_path)
                self.DS = data_npz["dynspec"]
                self.nu = data_npz["freq_MHz"]*MHz
                self.mjd = data_npz["time_MJD"]
                self.t = (self.mjd-self.mjd[0])*day
        else:
            #combine observations (currently only for standard encoding and same frequency range and time spacing)
            for i,obs in enumerate(date):
                data_path = file_path+'dynspec_'+obs+'.npz'
                data_npz = np.load(data_path)
                if i==0:
                    self.DS = data_npz["dynspec"]
                    self.nu = data_npz["freq_MHz"]*MHz
                    self.mjd = data_npz["time_MJD"]
                    dmjd = np.diff(self.mjd).mean()
                    N_nu = len(self.nu)
                else:
                    DS = data_npz["dynspec"]
                    mjd = data_npz["time_MJD"]
                    N_gap = int(np.rint((mjd[0]-self.mjd[-1])/dmjd))-1
                    mjd_gap = np.linspace(self.mjd[-1]+dmjd,self.mjd[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                    DS_gap = np.zeros((N_gap,N_nu),dtype=float)
                    mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                    self.DS = np.concatenate((self.DS,DS_gap,DS),axis=0)
                    self.mjd = np.concatenate((self.mjd,mjd_gap,mjd),axis=0)
            self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class B1508p55_LOFAR_classic(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508_LOFAR_2020/"
        data_path = file_path+'dynspec_LOFAR_'+date+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["I"]
        self.DS = self.DS/np.std(self.DS)
        print("Loading dynamic spectrum with dimensions {0}".format(self.DS.shape))
        self.nu = data_npz["nu_MHz"]*MHz
        self.mjd = data_npz["t_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class from_dsmaker2(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.DS /= np.std(self.DS)
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        self.mask = data_npz["mask"]
        self.pulse = data_npz["profile"]
        self.bpass = data_npz["bpass"]
        self.phase = np.linspace(0.,1.,num=self.pulse.shape[0])
        self.recalculate()
        
class from_ramain_lofar(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        data_npz = np.load(data_path)
        print(data_npz.files)
        self.DS = data_npz["dynspec"]
        self.DS /= np.std(self.DS)
        self.nu = data_npz["F"]*MHz
        self.mjd = data_npz["T"]
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class from_ramain_fits(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        f = fits.open(data_path)
        f.info()
        try:
            self.DS = np.copy(f[2].data)
            self.mjd = np.copy(f[4].data)
            self.nu = np.copy(f[5].data)*MHz
        except:
            self.DS = np.copy(f[1].data)
            self.nu = np.copy(f[2].data)*MHz
            self.mjd = np.copy(f[3].data)
        self.DS /= np.std(self.DS)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class from_ramain_lofar_gated(scinter_data.intensity):
    def __init__(self,data_path,gate):
        # - load data
        data_npz = np.load(data_path)
        print(data_npz.files)
        self.DS = data_npz["dyn{0}".format(gate)]
        self.DS /= np.std(self.DS)
        self.nu = data_npz["F"]*MHz
        self.mjd = data_npz["T"]
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class from_Parkes_npz(scinter_data.intensity):
    def __init__(self,data_path,mjd_init):
        # - load data
        data_npz = np.load(data_path,allow_pickle=True,fix_imports=True, encoding="latin1")
        print(data_npz.files)
        self.DS = data_npz["dynspec"]
        self.DS /= np.std(self.DS)
        self.nu = data_npz["F"]*MHz
        self.mjd = data_npz["T"]
        #print(self.mjd)
        #print(type(self.DS),self.DS.dtype,type(self.nu),self.nu.dtype,type(self.mjd),self.mjd.dtype)
        N_t = self.DS.shape[0]
        self.mjd = mjd_init + np.arange(N_t)*10./day
        self.t = (self.mjd-self.mjd[0])*day
        #self.mask = data_npz["mask"]
        #self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        self.recalculate()
        
class from_Parkes_dynspec(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        rawdata = np.loadtxt(data_path).transpose()
        with open(data_path, "r") as file:
            for line in file:
                if line.startswith("#"):
                    headline = str.strip(line[1:])
                    if str.split(headline) != []:
                        if str.split(headline)[0] == 'MJD0:':
                            # MJD of start of obs
                            mjd_init = float(str.split(headline)[1])
                            break
        self.t = np.unique(rawdata[2])*minute
        self.nu = np.unique(rawdata[3])*MHz
        self.mjd = mjd_init + self.t/day
        N_t = len(self.t)
        N_nu = len(self.nu)
        self.DS = rawdata[4].reshape((N_t,N_nu))
        self.DS /= np.std(self.DS)
        self.recalculate()
        
class B1508p55_Eff(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_'+date+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.DS /= np.std(self.DS)
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        #self.mask = data_npz["mask"]
        #self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        self.recalculate()
        
class B1508p55_Stinebring(scinter_data.intensity):
    def __init__(self,name):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508_Stinebring/Tim Sprenger - B1508+55/Tim Sprenger - B1508+55/"
        data_path = file_path+name+'.fits'
        f = fits.open(data_path)
        DS = f[0].data
        N_t = f[0].header["NAXIS2"]
        N_nu = f[0].header["NAXIS1"]
        mjd0 = f[0].header["MJD"]
        nu_min= f[0].header["CRVAL1"]*MHz
        t_min= f[0].header["CRVAL2"]
        dt = f[0].header["CDELT2"]
        dnu = f[0].header["CDELT1"]*MHz
        print(t_min,dt,N_t)
        print(nu_min,dnu,N_nu)
        
        self.DS = DS
        self.DS /= np.std(self.DS)
        self.nu = nu_min + np.arange(N_nu)*dnu
        self.t = t_min + np.arange(N_t)*dt
        self.mjd = mjd0 + self.t/day
        self.recalculate()
        
class B0834p06_brisken(scinter_data.intensity):
    def __init__(self,tel):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B0834+06_brisken/"
        data_path = file_path+'dynamic_spectrum_'+tel+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["I"]
        #self.DS /= np.std(self.DS)
        self.nu = data_npz["f_MHz"]*MHz
        self.t = data_npz["t_s"]
        #find more accurate date of observation!
        self.mjd = 53686. + self.t/day
        self.DS = self.DS/np.std(self.DS[self.DS!=0.])
        self.recalculate()
        
class B0834p06_brisken_vis(scinter_data.intensity):
    def __init__(self,tel):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B0834+06_brisken/"
        data_path = file_path+'dynamic_spectrum_'+tel+'.npz'
        data_npz = np.load(data_path)
        #print(data_npz["I"].shape,data_npz["I"].dtype,data_npz["I"][0,0])
        self.DS = np.nan_to_num(np.abs(data_npz["I"]))
        #print(np.mean(self.DS),np.std(self.DS),self.DS[0,0])
        self.phase = np.angle(data_npz["I"])
        self.phase = np.nan_to_num(self.phase)
        #print(np.mean(self.phase),np.std(self.phase),self.phase[0,0])
        #self.DS /= np.std(self.DS)
        self.nu = data_npz["f_MHz"]*MHz
        self.t = data_npz["t_s"]
        #find more accurate date of observation!
        self.mjd = 53686. + self.t/day
        self.DS = self.DS/np.std(self.DS[self.DS!=0.])
        self.recalculate()
        
class B1737p13_Stinebring(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1737+13/"
        data_path = file_path+'dynspec_B1737+13_'+date+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["I"]
        #self.DS /= np.std(self.DS)
        self.nu = data_npz["nu"]
        self.t = data_npz["t"]
        #find more accurate date of observation!
        self.mjd = data_npz["mjd0"] + self.t/day
        self.recalculate()
        
class B1508p55_Eff_paf(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_paf_'+date+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        #self.mask = data_npz["mask"]
        #self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        self.recalculate()
        
class B1508p55_Eff_edd(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        if type(date)==str:
            data_path = file_path+'raw_dynspec_Eff_edd_'+date+'.npz'
            data_npz = np.load(data_path)
            self.DS = data_npz["dynspec"]
            self.nu = data_npz["freq_MHz"]*MHz
            self.mjd = data_npz["time_MJD"]
            self.DS = self.DS/np.std(self.DS)
        else:
            #combine observations (currently only for standard encoding and same frequency range and time spacing)
            for i,obs in enumerate(date):
                data_path = file_path+'raw_dynspec_Eff_edd_'+obs+'.npz'
                data_npz = np.load(data_path)
                if i==0:
                    self.DS = data_npz["dynspec"]
                    self.nu = data_npz["freq_MHz"]*MHz
                    self.mjd = data_npz["time_MJD"]
                    dmjd = np.diff(self.mjd).mean()
                    N_nu = len(self.nu)
                    #- correct for scaling
                    self.DS = self.DS/np.std(self.DS)
                else:
                    DS = data_npz["dynspec"]
                    #- correct for scaling
                    DS = DS/np.std(DS)
                    mjd = data_npz["time_MJD"]
                    N_gap = int(np.rint((mjd[0]-self.mjd[-1])/dmjd))-1
                    if N_gap > 1:
                        mjd_gap = np.linspace(self.mjd[-1]+dmjd,self.mjd[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                        DS_gap = np.zeros((N_gap,N_nu),dtype=float)
                        mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                        self.DS = np.concatenate((self.DS,DS_gap,DS),axis=0)
                        self.mjd = np.concatenate((self.mjd,mjd_gap,mjd),axis=0)
                    else:
                        mjd = mjd + (dmjd-(mjd[0]-self.mjd[-1]))
                        self.DS = np.concatenate((self.DS,DS),axis=0)
                        self.mjd = np.concatenate((self.mjd,mjd),axis=0)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class B1508p55_Eff_edd_from_path(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.DS = self.DS/np.std(self.DS)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class B1508p55_Eff_edd_mask(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        if type(date)==str:
            data_path = file_path+'raw_dynspec_Eff_edd_'+date+'.npz'
            data_npz = np.load(data_path)
            self.DS = data_npz["mask"]
            self.nu = data_npz["freq_MHz"]*MHz
            self.mjd = data_npz["time_MJD"]
        else:
            #combine observations (currently only for standard encoding and same frequency range and time spacing)
            for i,obs in enumerate(date):
                data_path = file_path+'raw_dynspec_Eff_edd_'+obs+'.npz'
                data_npz = np.load(data_path)
                if i==0:
                    self.DS = data_npz["mask"]
                    self.nu = data_npz["freq_MHz"]*MHz
                    self.mjd = data_npz["time_MJD"]
                    dmjd = np.diff(self.mjd).mean()
                    N_nu = len(self.nu)
                else:
                    DS = data_npz["mask"]
                    #- correct for scaling
                    DS = DS/np.std(DS)
                    mjd = data_npz["time_MJD"]
                    N_gap = int(np.rint((mjd[0]-self.mjd[-1])/dmjd))-1
                    mjd_gap = np.linspace(self.mjd[-1]+dmjd,self.mjd[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                    DS_gap = np.zeros((N_gap,N_nu),dtype=float)
                    mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                    self.DS = np.concatenate((self.DS,DS_gap,DS),axis=0)
                    self.mjd = np.concatenate((self.mjd,mjd_gap,mjd),axis=0)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class B1508p55_Eff_UBB_low(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_'+date+'_UBB_low.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
       
class B1508p55_Eff_UBB_high(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_'+date+'_UBB_high.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class B1508p55_Eff_UBB(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        if type(date)==str:
            data_path = file_path+'raw_dynspec_Eff_'+date+'_UBB_low.npz'
            data_npz = np.load(data_path)
            self.DS = data_npz["dynspec"]
            self.nu = data_npz["freq_MHz"]*MHz
            self.mjd = data_npz["time_MJD"]
            self.t = (self.mjd-self.mjd[0])*day
            dnu = np.diff(self.nu).mean()
            N_t = len(self.t)
            # - combine with high frequency part
            data_path = file_path+'raw_dynspec_Eff_'+date+'_UBB_high.npz'
            data_npz = np.load(data_path)
            DS = data_npz["dynspec"]
            nu = data_npz["freq_MHz"]*MHz
            N_gap = int(np.rint((nu[0]-self.nu[-1])/dnu))-1
            if N_gap>0:
                nu_gap = np.linspace(self.nu[-1]+dnu,self.nu[-1]+N_gap*dnu,num=N_gap,endpoint=True)
                DS_gap = np.zeros((N_t,N_gap),dtype=float)
                nu = nu + (dnu-(nu[0]-nu_gap[-1]))
                self.DS = np.concatenate((self.DS,DS_gap,DS),axis=1)
                self.nu = np.concatenate((self.nu,nu_gap,nu),axis=0)
            else:
                self.DS = np.concatenate((self.DS,DS),axis=1)
                self.nu = np.concatenate((self.nu,nu),axis=0)
        else:
            #combine observations (currently only for standard encoding and same frequency range and time spacing)
            for i,obs in enumerate(date):
                data_path_low = file_path+'raw_dynspec_Eff_'+obs+'_UBB_low.npz'
                data_npz_low = np.load(data_path_low)
                data_path_high = file_path+'raw_dynspec_Eff_'+obs+'_UBB_high.npz'
                data_npz_high = np.load(data_path_high)
                if i==0:
                    DS_low = data_npz_low["dynspec"]
                    DS_high = data_npz_high["dynspec"]
                    #- normalize
                    #norm = np.mean(DS_low)
                    #DS_high *= norm/np.mean(DS_high)
                    DS_low /= np.std(DS_low)
                    DS_high /= np.std(DS_high)
                    nu_low = data_npz_low["freq_MHz"]*MHz
                    nu_high = data_npz_high["freq_MHz"]*MHz
                    mjd_low = data_npz_low["time_MJD"]
                    mjd_high = data_npz_high["time_MJD"]
                    dmjd = np.diff(mjd_low).mean()
                    N_nu_low = len(nu_low)
                    N_nu_high = len(nu_high)
                else:
                    DS = data_npz_low["dynspec"]
                    #DS *= norm/np.mean(DS)
                    DS /= np.std(DS)
                    mjd = data_npz_low["time_MJD"]
                    N_gap = int(np.rint((mjd[0]-mjd_low[-1])/dmjd))-1
                    if 0:#N_gap>0:
                        mjd_gap = np.linspace(mjd_low[-1]+dmjd,mjd_low[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                        DS_gap = np.zeros((N_gap,N_nu_low),dtype=float)
                        mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                        DS_low = np.concatenate((DS_low,DS_gap,DS),axis=0)
                        mjd_low = np.concatenate((mjd_low,mjd_gap,mjd),axis=0)
                    else:
                        DS_low = np.concatenate((DS_low,DS),axis=0)
                        mjd_low = np.concatenate((mjd_low,mjd),axis=0)
                    
                    DS = data_npz_high["dynspec"]
                    #DS *= norm/np.mean(DS)
                    DS /= np.std(DS)
                    mjd = data_npz_high["time_MJD"]
                    N_gap = int(np.rint((mjd[0]-mjd_high[-1])/dmjd))-1
                    #print((mjd[-1]-mjd_low[-1])*day)
                    if 0:#N_gap>0:
                        mjd_gap = np.linspace(mjd_high[-1]+dmjd,mjd_high[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                        DS_gap = np.zeros((N_gap,N_nu_high),dtype=float)
                        mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                        DS_high = np.concatenate((DS_high,DS_gap,DS),axis=0)
                        mjd_high = np.concatenate((mjd_high,mjd_gap,mjd),axis=0)
                    else:
                        DS_high = np.concatenate((DS_high,DS),axis=0)
                        mjd_high = np.concatenate((mjd_high,mjd),axis=0)
            # - fixing different lengths
            DS_high = np.concatenate((DS_high,np.zeros((1,N_nu_high),dtype=float)),axis=0)
            print(mjd_low[-1]*day,mjd_high[-1]*day)
            self.DS = np.concatenate((DS_low,DS_high),axis=1)
            self.nu = np.concatenate((nu_low,nu_high),axis=0)
            self.mjd = np.linspace(mjd_low[0],mjd_low[-1],num=len(mjd_low),endpoint=True)
            self.t = (self.mjd-self.mjd[0])*day
        
        self.recalculate()


        
class B1508p55_Eff_pol(scinter_data.intensity):
    def __init__(self,date,i_pol):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_pol'+str(i_pol)+'_dynspec_Eff_'+date+'.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        #self.mask = data_npz["mask"]
        #self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        self.recalculate()
        
class B1508p55_Eff_pulse_paf(scinter_data.pulse_profile):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_paf_'+date+'.npz'
        data_npz = np.load(data_path)
        #self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        #self.mjd = data_npz["time_MJD"]
        #self.mask = data_npz["mask"]
        self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        period = 0.739681265668
        self.t = np.linspace(0.,period,num=self.pulse.shape[0])
        self.recalculate()
        
class B1508p55_Eff_pulse(scinter_data.pulse_profile):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_'+date+'.npz'
        data_npz = np.load(data_path)
        #self.DS = data_npz["dynspec"]
        self.nu = data_npz["freq_MHz"]*MHz
        #self.mjd = data_npz["time_MJD"]
        #self.mask = data_npz["mask"]
        self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        period = 0.739681265668
        self.t = np.linspace(0.,period,num=self.pulse.shape[0])
        self.recalculate()
        
class B1508p55_Jiamusi(scinter_data.intensity):
    def __init__(self,fig):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508_Jiamusi/"
        #freq,time,amp = np.genfromtxt(file_path+fig)
        freq,time,amp = np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)
        with open(file_path+fig,'r') as readfile:
            for line in readfile.readlines():
                if line != "\n":
                    entries = line.split()
                    freq = np.append(freq,float(entries[0]))
                    time = np.append(time,float(entries[1]))
                    amp = np.append(amp,float(entries[2]))
            
        #time = time[~np.isnan(time)]
        #amp = amp[~np.isnan(amp)]
        
        N_nu = 256
        self.nu = freq[:N_nu]*MHz
        self.mjd = time[::N_nu]
        self.t = (self.mjd-self.mjd[0])*day
        N_t = len(self.t)
        #self.DS = np.swapaxes(amp.reshape(N_nu,N_t),0,1)
        self.DS = amp.reshape(N_t,N_nu)
        self.recalculate()
        
class B1508p55_FAST(scinter_data.intensity):
    def __init__(self,date):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_FAST/B1508+55_{0}.ds".format(date)
        if 0:
            i_t,i_nu,time,freq,amp = np.array([],dtype=int),np.array([],dtype=int),np.array([],dtype=float),np.array([],dtype=float),np.array([],dtype=float)
            with open(file_path,'r') as readfile:
                for line in readfile.readlines():
                    if line != "\n":
                        entries = line.split()
                        if entries[0]=="#":
                            MJD0 = float(entries[-1])
                            print("MJD0 = {0}".format(MJD0))
                        else:
                            i_t = np.append(i_t,float(entries[0]))
                            i_nu = np.append(i_nu,float(entries[1]))
                            time = np.append(time,float(entries[2]))
                            freq = np.append(freq,float(entries[3]))
                            amp = np.append(amp,float(entries[4]))
        else:
            with open(file_path,'r') as readfile:
                lines = readfile.readlines()
                entries = lines[0].split()
                MJD0 = float(entries[-1])
                print("MJD0 = {0}".format(MJD0))
            i_t,i_nu,time,freq,amp,err = np.genfromtxt(file_path,unpack=True,skip_header=1)
            
        #time = time[~np.isnan(time)]
        #amp = amp[~np.isnan(amp)]
        
        N_nu = int(np.rint(np.max(i_nu)+1))
        N_t = int(np.rint(np.max(i_t)+1))
        self.nu = freq[:N_nu]*MHz
        self.t = time[::N_nu]*minute
        self.mjd = MJD0 + self.t/day
        #self.DS = np.swapaxes(amp.reshape(N_nu,N_t),0,1)
        self.DS = amp.reshape(N_t,N_nu)
        self.recalculate()
        
class B1508p55_Eff_classic_phased(scinter_data.intensity):
    def __init__(self,date,i_phase):
        # - load data
        file_path = "/mnt/d/Ubuntu/MPIfR/keeper/Seafile/data/B1508+55_spectra_filtered/"
        data_path = file_path+'raw_dynspec_Eff_'+date+'_phased.npz'
        data_npz = np.load(data_path)
        self.DS = data_npz["dynspec"][:,:,i_phase]
        self.nu = data_npz["freq_MHz"]*MHz
        self.mjd = data_npz["time_MJD"]
        self.t = (self.mjd-self.mjd[0])*day
        #self.mask = data_npz["mask"]
        #self.pulse = data_npz["profile"]
        #self.bpass = data_npz["bpass"]
        self.recalculate()
        
class from_npz(scinter_data.intensity):
    def __init__(self,data_path):
        # - load data
        data_npz = np.load(data_path)
        self.DS = data_npz["DS"]
        self.nu = data_npz["nu"]
        self.mjd = data_npz["mjd"]
        self.DS = self.DS/np.std(self.DS)
        self.t = (self.mjd-self.mjd[0])*day
        self.recalculate()
        
class Geetam_data(scinter_data.intensity):
    def __init__(self,data_path):
        if type(data_path)==str:
            # - load data
            data_npz = np.load(data_path)
            self.DS = data_npz["dynspec"]
            self.nu = data_npz["F"]*MHz
            self.mjd = data_npz["t"]
            self.DS = self.DS/np.std(self.DS)
            self.t = (self.mjd-self.mjd[0])*day
        else:
            #combine observations (currently only for standard encoding and same frequency range and time spacing)
            for i,file_path in enumerate(data_path):
                data_npz = np.load(file_path)
                if i==0:
                    self.DS = data_npz["dynspec"]
                    self.nu = data_npz["F"]*MHz
                    self.mjd = data_npz["t"]
                    dmjd = np.diff(self.mjd).mean()
                    N_nu = len(self.nu)
                else:
                    DS = data_npz["dynspec"]
                    mjd = data_npz["t"]
                    N_gap = int(np.rint((mjd[0]-self.mjd[-1])/dmjd))-1
                    mjd_gap = np.linspace(self.mjd[-1]+dmjd,self.mjd[-1]+N_gap*dmjd,num=N_gap,endpoint=True)
                    DS_gap = np.zeros((N_gap,N_nu),dtype=float)
                    mjd = mjd + (dmjd-(mjd[0]-mjd_gap[-1]))
                    self.DS = np.concatenate((self.DS,DS_gap,DS),axis=0)
                    self.mjd = np.concatenate((self.mjd,mjd_gap,mjd),axis=0)
            self.t = (self.mjd-self.mjd[0])*day
        self.DS = np.nan_to_num(self.DS)
        self.DS = self.DS/np.std(self.DS)
        self.recalculate()
