import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from astropy.coordinates import SkyCoord

"""
set up paths:
scinter3_path : '/PATH_TO_YOUR/Scinter3'
mem_path : "/DIRECTORY_TO_SAVE_PLOTS"

The usage of scinter_plot and scinter_data is independent of the simulations and can be replaced by other plotting and data analysis functions without conflict.
"""

sys.path.insert(0,scinter3_path)
import scinter_data
import scinter_plot
import scinter_computation

#reproducible random values
rng = np.random.default_rng(12345)

#some constants in SI units
degrees = np.pi/180.
mas = degrees/1000./3600.
hour = 3600.
year = 365.25*24.*hour
au = 149597870700. #m
pc = 648000./np.pi*au #m
v_c = 299792458.
minute = 60.
day = 24.*hour
mHz = 1.0e-3
MHz = 1.0e+6
GHz = 1.0e+9
mus = 1.0e-6
sqrtmus = np.sqrt(mus)
kms = 1000.
vm_scale = hour/sqrtmus

##### pulsar model #####
print("Setting up pulsar model ...")

"""
define pulsar parameters:
psrname : name of the pulsar readable in astropy
PMRA : proper motion along right ascension in milliarcseconds per year
PMDEC : proper motion along declination in milliarcseconds per year
D_s : pulsar distance in parsec
"""

##### data model #####
print("Setting up observation series ...")

"""
define the range in time where the variation of observables shall be predicted:
mjds : modified Julian dates to evaluate model on
"""

screen = scinter_computation.Evolution_Two1DScreens(mjds,psrname)

##### data model #####
print("Setting up geometry of screens ...")

"""
define geometric parameters of the two screens (first screen is the closer one):
D_x : distance of first screen in parsec
a_x : angle of first screen in degrees 
V_x : ISM velocity along the first screen in km/s
D_y : distance of second screen in parsec
a_y : angle of second screen in degrees 
V_y : ISM velocity along the second screen in km/s
"""

SP = scinter_computation.Two1DScreens(D_x=D_x,D_y=D_y,D_s=D_s,a_x=a_x,a_y=a_y,V_x=V_x,V_y=V_y,PMRA=PMRA,PMDEC=PMDEC)

##### image model #####
print("Setting up images on the screens ...")

"""
Use SP.add_SPx(th=th,mu=mu,ph=ph) to add a point on the first screen:
th : angular position in radians
mu : amplitude in arbitrary units
ph : dispersive phase in radians
Use SP.add_SPy(th=th,mu=mu,ph=ph) to add a point on the second screen.
Use SP.add_CPx(mu=mu) and SP.add_CPy(mu=mu) to add a single comoving central point.
"""

print("Plotting screens ...")

th1_x = SP.x/SP.D_x*np.cos(a_x*degrees)/mas
th1_y = SP.x/SP.D_x*np.sin(a_x*degrees)/mas
th2_x = SP.y/SP.D_y*np.cos(a_y*degrees)/mas
th2_y = SP.y/SP.D_y*np.sin(a_y*degrees)/mas

##### data model #####
print("Setting up simulated data ...")

"""
define coordinates to evaluate the simulation on:
t : 1D array of times in seconds starting at 0
mjds_data : corresponding modified Julian dates
nu: 1D array of frequencies in HZ
"""

#random noise in E-field
noise_rms = 0.1
#strength of random pulse-to-pulse variation
noise_pulse = 0.

SP.add_noise(noise_rms=noise_rms)
SP.add_pulse_variation(noise_pulse=noise_pulse)

##### predict evolution of parameters #####
print("Computing evolution ...")

"""
Add real data to plot to compare with predictions.
"""

def eta_from_zeta(zeta):
    return 1./(2.*1.4*GHz*zeta)**2

zetas = screen.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,D_y=D_y,a_x=a_x,a_y=a_y,V_x=V_x,V_y=V_y)

figure = scinter_plot.draw_canvas(plot_width = 1600,plot_height = 900, plot_bottom = 0.07, plot_left = 0.12, plot_top = 0.97, plot_right=0.7, plot_wspace = 0.2, plot_hspace = 0.2, textsize=14, labelsize=12)
plt.grid(alpha=0.5,linestyle='--')
ax = figure.add_subplot(1,1,1)

P2x, = ax.plot(mjds,eta_from_zeta(zetas['zeta2_x']),linestyle='-',markersize=0,color='blue',label="2scr 1st arc")
P2y, = ax.plot(mjds,eta_from_zeta(zetas['zeta2_y']),linestyle='-',markersize=0,color='red',label="2scr 2nd arc")
P1x, = ax.plot(mjds,eta_from_zeta(zetas['zeta1_x']),linestyle='--',markersize=0,color='blue',label="1scr 1st arc")
P1y, = ax.plot(mjds,eta_from_zeta(zetas['zeta1_y']),linestyle='--',markersize=0,color='red',label="1scr 2nd arc")

plt.legend(loc='best')
ax.set_xlabel("MJD")
ax.set_ylabel("r$\eta$ [s$^3$]")

figure.savefig(os.path.join(mem_path,"curvature_evolution.png"))
plt.close()

##### predict evolution of parameters #####
print("Simulating dynamic spectrum ...")

"""
If it takes long, consider to save the intermediate result.
"""

E, dynspec = SP.compute_DS(t,nu,mjds_data,psrname)

##### predict evolution of parameters #####
print("Plotting dynamic spectrum ...")

#transfer simulations to scinter_data object
DS = scinter_data.generic_intensity(t,nu,dynspec,np.mean(mjds_data))

figure = scinter_plot.draw_canvas(plot_width = 1200,plot_height = 700, plot_bottom = 0.1, plot_left = 0.1, plot_top=0.92, plot_right=0.96, plot_wspace = 0.3, plot_hspace = 0.3, textsize=16, labelsize=12)
ax = figure.add_subplot(1,1,1)

scinter_plot.dynamic_spectrum(DS.t,DS.nu,DS.DS,ax,vmin=None,vmax=None,title="simulated dynamic spectrum",nu_sampling=2)

figure.savefig(os.path.join(mem_path,"dynspec.png"))
plt.close()

##### secondary spectrum in transforms #####
print("Computing secondary spectrum ...")

Lmb = scinter_data.SecSpec_Lambda(DS)
FFT = scinter_data.SecSpec_FFT(DS)
NuT = scinter_data.SecSpec_NuT(DS,nu0=DS.nu0)

print("Plotting secondary spectrum ...")

figure = scinter_plot.draw_canvas(plot_width = 2400,plot_height = 700, plot_bottom = 0.1, plot_left = 0.1, plot_wspace = 0.1, plot_hspace = 0.1, textsize=16, labelsize=12)
ax_FFT = figure.add_subplot(1,3,1)
ax_NuT = figure.add_subplot(1,3,2)
ax_Lmb = figure.add_subplot(1,3,3)

fD_cut = np.max(FFT.fD)/mHz
tau_cut = np.max(FFT.tau)/mus
tau_sampling = 10
fD_sampling = 1
vmin = None
vmax = None
plot1 = scinter_plot.secondary_spectrum(FFT.fD,FFT.tau,FFT.SS,ax_FFT,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_cut,fD_max=fD_cut,tau_min=-tau_cut,tau_max=tau_cut,vmin=vmin,vmax=vmax,title="FFT")
figure.colorbar(plot1, ax=ax_FFT)
plot2 = scinter_plot.secondary_spectrum(NuT.fD,NuT.tau,NuT.SS,ax_NuT,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_cut,fD_max=fD_cut,tau_min=-tau_cut,tau_max=tau_cut,vmin=vmin,vmax=vmax,title="NuT")
figure.colorbar(plot2, ax=ax_NuT)
plot3 = scinter_plot.secondary_spectrum(Lmb.fD,Lmb.tau,Lmb.SS,ax_Lmb,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_cut,fD_max=fD_cut,tau_min=-tau_cut,tau_max=tau_cut,vmin=vmin,vmax=vmax,title="Lmb")
figure.colorbar(plot3, ax=ax_Lmb)

figure.savefig(os.path.join(mem_path,"secspec.png"))
plt.close()

print("Computing wavefield ...")

EF = scinter_data.generic_Efield(t,nu,E)
WF = scinter_data.Wavefield_NuT(DS,EF)

print("Plotting wavefield ...")

figure = scinter_plot.draw_canvas(plot_width = 1200,plot_height = 600, plot_bottom = 0.1, plot_left = 0.1, plot_top=0.92, plot_wspace = 0.3, plot_hspace = 0.3, textsize=16, labelsize=12)

ax1 = figure.add_subplot(1,2,1)
ax2 = figure.add_subplot(1,2,2)

vmin_E = None
vmax_E = None

scinter_plot.secondary_spectrum(NuT.fD,NuT.tau,NuT.SS,ax1,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_cut,fD_max=fD_cut,tau_min=-tau_cut,tau_max=tau_cut,vmin=vmin,vmax=vmax,title="secondary spectrum")
scinter_plot.secondary_spectrum(WF.fD,WF.tau,WF.amplitude,ax2,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_cut,fD_max=fD_cut,tau_min=-tau_cut,tau_max=tau_cut,vmin=vmin_E,vmax=vmax_E,title="wavefield")

figure.savefig(os.path.join(mem_path,"wavefield.png"))
plt.close()