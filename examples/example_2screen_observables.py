import numpy as np
from astropy.coordinates import EarthLocation, Angle
import matplotlib as mpl
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
import sys

"""
set up paths:
scinter3_path : '/PATH_TO_YOUR/Scinter3'

The usage of scinter_plot is independent of the simulations and can be replaced by other plotting and data analysis functions without conflict.
"""
sys.path.insert(0,scinter3_path)
import scinter_computation
import scinter_plot

#some constants in SI units
mus = 1.0e-6
sqrtmus = np.sqrt(mus)
hour = 3600.
vm_scale = hour/sqrtmus

### set up initial values
"""
define pulsar parameters:
psrname : name of the pulsar readable in astropy
PMRA : proper motion along right ascension in milliarcseconds per year
PMDEC : proper motion along declination in milliarcseconds per year
D_s : pulsar distance in parsec
"""
D_s = 2100.
PMRA = -73.64
PMDEC = -62.65
telescope = EarthLocation.from_geodetic(Angle('6°52′58″'),Angle('50°31′29″'),height=319.) #Effelsberg
psrname = 'PSRB1508+55'

"""
define geometric parameters of the two screens (first screen is the closer one):
D_x : distance of first screen in parsec
a_x : angle of first screen in degrees 
V_x : absolute value ISM velocity within first screen in km/s
a_Vx: orientation of ISM velocity within first screen in degrees
D_y : distance of second screen in parsec
a_y : angle of second screen in degrees 
V_y : ISM velocity along the second screen in km/s
"""
D_x = 127.1
D_y = 1935
a_x = 37.58
a_y = -39.7
V_x = 7.3
V_y = -6.
a_Vx = -11.

"""
define the range in time where the variation of observables shall be predicted:
mjds : modified Julian dates to evaluate model on
"""
mjds = np.linspace(60000.,61000.,num=100)

### compute the velocity of the telescope at the observation times
screen = scinter_computation.Evolution_Two1DScreens(mjds,psrname,telcoords=telescope)
"""
Options:
include_earth_rotation_in_veff : apply correction to velocity from Earth's rotation (careful: varies during observation and is already included in p(t) where it is used) default: False
"""
#compute observables from initial values
V_x_par = V_x*np.cos((a_x-a_Vx)*np.pi/180.) #velocity along first screen
zetas = screen.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,a_x=a_x,V_x=V_x_par,a_y=a_y,D_y=D_y,V_y=V_y)
"""
returns a dictionary of predicted observables (N_t = length of mjds)
Deff1_x : effective distance along first screen in one-screen theory | scalar
Deff1_y : effective distance along second screen in one-screen theory | scalar
Deff2_x : effective distance along first screen in two-screen theory | scalar
Deff2_y : effective distance along second screen in two-screen theory | scalar
D_mix   : distance factor of mixed term in two-screen theory | scalar
Veff1_x : effective velocity along first screen in one-screen theory | (N_t)-array
Veff1_y : effective velocity along second screen in one-screen theory | (N_t)-array
Veff2_x : effective velocity along first screen in two-screen theory | (N_t)-array
Veff2_y : effective velocity along second screen in two-screen theory | (N_t)-array
Veff1_x_vec : effective velocity vector of first screen in one-screen theory | (N_t,2)-array
zeta1_x : zeta from arc curvature of first screen in one-screen theory | (N_t)-array
zeta1_y : zeta from arc curvature of second screen in one-screen theory | (N_t)-array
zeta2_x : zeta from arc curvature of first screen in two-screen theory | (N_t)-array
zeta2_y : zeta from arc curvature of second screen in two-screen theory | (N_t)-array
zeta2_m : zeta from modulation speed (modulation of arc of first screen by dynamic spectrum of second screen) | (N_t)-array
zeta2_fx: zeta from feature movement of first screen in two-screen theory (equal to that of arc curvature in one-screen theory) | (N_t)-array
zeta2_fy: zeta from feature movement of second screen in two-screen theory (equal to that of arc curvature in one-screen theory) | (N_t)-array
Dt_2x   : delay of scintillation pattern with respect to center of Earth due to first screen in two-screen theory | (N_t)-array
Dt_2y   : delay of scintillation pattern with respect to center of Earth due to second screen in two-screen theory | (N_t)-array
"""

#If required, zeta can be translated to an arc curvature at frequency nu0 in Hertz with
def eta_from_zeta(zeta,nu0):
    return 1./(2.*nu0*zeta)**2

### example interactive plot showing the zetas of arc curvatures of both screens in both one-screen (radiation is unaffected by one of the screens) and two-screen (scattered at both screens) theory
figure = scinter_plot.draw_canvas(plot_width = 1600,plot_height = 900, plot_bottom = 0.07, plot_left = 0.12, plot_top = 0.97, plot_right=0.7, plot_wspace = 0.2, plot_hspace = 0.2, textsize=14, labelsize=12)
ax = figure.add_subplot(1,1,1)
ax.set_ylabel(r"$\partial_t \sqrt{\tau}$   [$\sqrt{\mu s}$/h]")
ax.set_xlabel("MJD")
ax.set_ylim([0.,0.06])

plt.grid(alpha=0.5,linestyle='--')

plot_zeta2_x, = ax.plot(mjds,zetas['zeta2_x']*vm_scale,linestyle='-',markersize=0,color='blue',label="1st screen (interaction)")
plot_zeta2_y, = ax.plot(mjds,zetas['zeta2_y']*vm_scale,linestyle='-',markersize=0,color='red',label="2nd screen (interaction)")
plot_zeta1_x, = ax.plot(mjds,zetas['zeta1_x']*vm_scale,linestyle='--',markersize=0,color='blue',label="1st screen (direct)")
plot_zeta1_y, = ax.plot(mjds,zetas['zeta1_y']*vm_scale,linestyle='--',markersize=0,color='red',label="2nd screen (direct)")

#make sliders
slider_V_y = mpl.widgets.Slider(plt.axes([0.75,0.1,0.2,0.06]),r'$V_{y}$ [km/s]',0.,150.,valinit=V_y)
slider_V_x = mpl.widgets.Slider(plt.axes([0.75,0.2,0.2,0.06]),r'$V_{x}$ [km/s]',0.,150.,valinit=V_x)
slider_a_y = mpl.widgets.Slider(plt.axes([0.75,0.3,0.2,0.06]),r'$\alpha_y$ [$\circ$]',-180.,180.,valinit=a_y)
slider_a_Vx = mpl.widgets.Slider(plt.axes([0.75,0.4,0.2,0.06]),r'$\alpha_{V_x}$ [$\circ$]',-180.,180.,valinit=a_Vx)
slider_a_x = mpl.widgets.Slider(plt.axes([0.75,0.5,0.2,0.06]),r'$\alpha_x$ [$\circ$]',-180.,180.,valinit=a_x)
slider_D_y = mpl.widgets.Slider(plt.axes([0.75,0.7,0.2,0.06]),r'$d_{y}$ [pc]',90.,2100.,valinit=D_y)
slider_D_x = mpl.widgets.Slider(plt.axes([0.75,0.8,0.2,0.06]),r'$d_{x}$ [pc]',50.,200.,valinit=D_x)

def update_plot(event):
	a_x = slider_a_x.val
	a_Vx = slider_a_Vx.val
	D_x = slider_D_x.val
	V_x = slider_V_x.val
	a_y = slider_a_y.val
	D_y = slider_D_y.val
	V_y = slider_V_y.val
	
	V_x_par = V_x*np.cos((a_x-a_Vx)*np.pi/180.)
	zetas = screen.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,D_y=D_y,a_x=a_x,a_y=a_y,V_x=V_x_par,V_y=V_y)
	
	plot_zeta2_x.set_ydata(zetas['zeta2_x']*vm_scale)
	plot_zeta2_y.set_ydata(zetas['zeta2_y']*vm_scale)
	plot_zeta1_x.set_ydata(zetas['zeta1_x']*vm_scale)
	plot_zeta1_y.set_ydata(zetas['zeta1_y']*vm_scale)
	
	figure.canvas.draw_idle()

slider_a_x.on_changed(update_plot)
slider_a_Vx.on_changed(update_plot)
slider_D_x.on_changed(update_plot)
slider_V_x.on_changed(update_plot)
slider_a_y.on_changed(update_plot)
slider_D_y.on_changed(update_plot)
slider_V_y.on_changed(update_plot)

ax.legend()

plt.show()
