import numpy as np
from numpy import newaxis as na
import matplotlib as mpl
#import matplotlib.style as mplstyle
#mplstyle.use('fast')
import matplotlib.pyplot as plt
import skimage
from skimage.measure import block_reduce
from PIL import Image
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
e = 1.602176634e-19 #C
me = 9.1093837015e-31 #kg
eps0 = 8.8541878128e-12 #SI
pc_per_cm3 = pc/0.01**3
hour = 3600.
minute = 60.
kms = 1000.

def draw_canvas(plot_width = 1200, plot_height = 900, plot_dpi = 100, plot_bottom = 0.08, plot_top = 0.95, plot_left = 0.08, plot_right = 0.98, plot_wspace = 0.1, plot_hspace = 0.2, textsize=8, labelsize=6):
    figure = plt.figure(figsize=(plot_width/plot_dpi,plot_height/plot_dpi),dpi=plot_dpi)
    plt.subplots_adjust(bottom=plot_bottom,top=plot_top,left=plot_left,right=plot_right,wspace=plot_wspace,hspace=plot_hspace)
    pgf_with_pdflatex = {
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.size": labelsize,
        "axes.linewidth": 0.5,                
        "axes.labelsize": labelsize,               # LaTeX default is 10pt font. 
        "axes.titlesize": labelsize,
        "patch.linewidth": 0.5,		# Width of box around legend
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "lines.markeredgewidth": 0.3,
        "legend.fontsize": textsize,
        "legend.edgecolor": "black",
        "legend.borderpad": 0.3,			# width of whitespace between text and border (in units of times linewidth)
        "xtick.labelsize": textsize,
        "ytick.labelsize": textsize,
    }
    mpl.rcParams.update(pgf_with_pdflatex) 
    
    return figure
    

def colormesh(x,y,f_xy,ax,x_sampling=1,y_sampling=1,cmap='viridis',vmin=None,vmax=None,log10=False):
    #Preprocess data
    # - downsampling
    f_xy = block_reduce(f_xy, block_size=(x_sampling,y_sampling), func=np.mean)
    coordinates = np.array([x,x])
    coordinates = block_reduce(coordinates, block_size=(1,x_sampling), func=np.mean, cval=x[-1])
    x = coordinates[0,:]
    coordinates = np.array([y,y])
    coordinates = block_reduce(coordinates, block_size=(1,y_sampling), func=np.mean, cval=y[-1])
    y = coordinates[0,:]
    # # - compute offsets to center pccolormesh
    # offset_x = 0. #(x[1]-x[0])/2.
    # offset_y = 0. #(y[1]-y[0])/2.
    if log10:
        f_xy[f_xy < 0.] = 0.
        if vmin==None:
            min_nonzero = np.min(f_xy[np.nonzero(f_xy)])
            f_xy[f_xy == 0.] = min_nonzero
        else:
            f_xy[f_xy == 0.] = 10**vmin
        f_xy = np.log10(f_xy)
    
    #draw the plot
    #im = ax.pcolormesh((x-offset_x),(y-offset_y),np.swapaxes(f_xy,0,1),cmap=cmap,vmin=vmin,vmax=vmax)
    im = ax.pcolormesh(x,y,np.swapaxes(f_xy,0,1),cmap=cmap,vmin=vmin,vmax=vmax,shading='nearest')
    
    return im
    
def dynamic_spectrum(t,nu,DS,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Dynamic Spectrum")
    xlabel = kwargs.get("xlabel",r"$t$ [min]")
    ylabel = kwargs.get("ylabel",r"$\nu$ [MHz]")
    xscale = kwargs.get("xscale",minute)
    yscale = kwargs.get("yscale",MHz)
    t_min = kwargs.get("t_min",np.min(t)/xscale)
    t_max = kwargs.get("t_max",np.max(t)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/yscale)
    
    #draw the plot
    im = colormesh(t/xscale,nu/yscale,DS,ax,x_sampling=t_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([t_min,t_max])
    ax.set_ylim([nu_min,nu_max])
    
    return im
    
def pulse_profile(t,nu,pulse,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Pulse Profile")
    xlabel = kwargs.get("xlabel",r"$t$ [s]")
    ylabel = kwargs.get("ylabel",r"$\nu$ [MHz]")
    xscale = kwargs.get("xscale",1.)
    yscale = kwargs.get("yscale",MHz)
    t_min = kwargs.get("t_min",np.min(t)/xscale)
    t_max = kwargs.get("t_max",np.max(t)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/yscale)
    log10 = kwargs.get("log10",False)
    
    #draw the plot
    im = colormesh(t/xscale,nu/yscale,pulse,ax,x_sampling=t_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([t_min,t_max])
    ax.set_ylim([nu_min,nu_max])
    
    return im
    
def secondary_spectrum(fD,tau,SS,ax,**kwargs):
    #parameters
    fD_sampling = kwargs.get("fD_sampling",1)
    tau_sampling = kwargs.get("tau_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Secondary Spectrum")
    SS_type = kwargs.get("type","frequency")
    if SS_type == "frequency":
        xlabel = kwargs.get("xlabel",r"$f_D$ [mHz]")
        ylabel = kwargs.get("ylabel",r"$\tau$ [$\mu$s]")
        xscale = kwargs.get("xscale",mHz)
        yscale = kwargs.get("yscale",musec)
        fD_min = kwargs.get("fD_min",np.min(fD)/xscale)
        fD_max = kwargs.get("fD_max",np.max(fD)/xscale)
        tau_min = kwargs.get("tau_min",np.min(tau)/yscale)
        tau_max = kwargs.get("tau_max",np.max(tau)/yscale)
        
    elif SS_type == "wavelength":
        xlabel = kwargs.get("xlabel",r"$f_D$ [mHz]")
        ylabel = kwargs.get("ylabel",r"$Beta$ [m$^{-1}$]")
        xscale = kwargs.get("xscale",mHz)
        yscale = kwargs.get("yscale",1)
        fD_min = kwargs.get("fD_min",np.min(fD)/xscale)
        fD_max = kwargs.get("fD_max",np.max(fD)/xscale)
        tau_min = kwargs.get("tau_min",np.min(tau)/yscale)
        tau_max = kwargs.get("tau_max",np.max(tau)/yscale)
        
    #draw the plot
    im = colormesh(fD/xscale,tau/yscale,SS,ax,x_sampling=fD_sampling,y_sampling=tau_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([fD_min,fD_max])
    ax.set_ylim([tau_min,tau_max])
    
    return im

def ttau_spectrum(t,tau,hSS,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    tau_sampling = kwargs.get("tau_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"FFT over frequency")
    xlabel = kwargs.get("xlabel",r"$t$ [min]")
    ylabel = kwargs.get("ylabel",r"$\tau$ [$\mu$s]")
    xscale = kwargs.get("xscale",minute)
    yscale = kwargs.get("yscale",musec)
    t_min = kwargs.get("t_min",np.min(t)/xscale)
    t_max = kwargs.get("t_max",np.max(t)/xscale)
    tau_min = kwargs.get("tau_min",np.min(tau)/yscale)
    tau_max = kwargs.get("tau_max",np.max(tau)/yscale)
    
    #draw the plot
    im = colormesh(t/xscale,tau/yscale,hSS,ax,x_sampling=t_sampling,y_sampling=tau_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([t_min,t_max])
    ax.set_ylim([tau_min,tau_max])
    
    return im
    
def dynamic_phase(t,nu,Ph,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'twilight')
    vmin = kwargs.get("vmin",float(-np.pi))
    vmax = kwargs.get("vmax",float(np.pi))
    title = kwargs.get("title",r"Phase")
    xlabel = kwargs.get("xlabel",r"$t$ [s]")
    ylabel = kwargs.get("ylabel",r"$\nu$ [MHz]")
    xscale = kwargs.get("xscale",1.)
    yscale = kwargs.get("yscale",MHz)
    t_min = kwargs.get("t_min",np.min(t)/xscale)
    t_max = kwargs.get("t_max",np.max(t)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/yscale)
    
    #draw the plot
    im = colormesh(t/xscale,nu/yscale,Ph,ax,x_sampling=t_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([t_min,t_max])
    ax.set_ylim([nu_min,nu_max])
    
    return im
    
def secondary_phase(fD,tau,Ph,ax,**kwargs):
    #parameters
    fD_sampling = kwargs.get("fD_sampling",1)
    tau_sampling = kwargs.get("tau_sampling",1)
    cmap = kwargs.get("cmap",'twilight')
    vmin = kwargs.get("vmin",float(-np.pi))
    vmax = kwargs.get("vmax",float(np.pi))
    title = kwargs.get("title",r"Phase")
    xlabel = kwargs.get("xlabel",r"$f_D$ [mHz]")
    ylabel = kwargs.get("ylabel",r"$\tau$ [$\mu$s]")
    xscale = kwargs.get("xscale",mHz)
    yscale = kwargs.get("yscale",musec)
    fD_min = kwargs.get("fD_min",np.min(fD)/xscale)
    fD_max = kwargs.get("fD_max",np.max(fD)/xscale)
    tau_min = kwargs.get("tau_min",np.min(tau)/yscale)
    tau_max = kwargs.get("tau_max",np.max(tau)/yscale)
    
    #draw the plot
    im = colormesh(fD/xscale,tau/yscale,Ph,ax,x_sampling=fD_sampling,y_sampling=tau_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([fD_min,fD_max])
    ax.set_ylim([tau_min,tau_max])
    
    return im
    
def staufD_diagram(fD,stau,staufD,ax,**kwargs):
    #parameters
    fD_sampling = kwargs.get("fD_sampling",1)
    stau_sampling = kwargs.get("stau_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"$f_D$-$\sqrt{\tau}$-Diagram")
    xlabel = kwargs.get("xlabel",r"$f_D$ [mHz]")
    ylabel = kwargs.get("ylabel",r"$\sqrt{\tau}$ [$\mu$s$^{1/2}$]")
    xscale = kwargs.get("xscale",mHz)
    yscale = kwargs.get("yscale",np.sqrt(musec))
    fD_min = kwargs.get("fD_min",np.min(fD)/xscale)
    fD_max = kwargs.get("fD_max",np.max(fD)/xscale)
    stau_min = kwargs.get("stau_min",np.min(stau)/yscale)
    stau_max = kwargs.get("stau_max",np.max(stau)/yscale)
    log10 = kwargs.get("log10",True)
    
    #draw the plot
    im = colormesh(fD/xscale,stau/yscale,staufD,ax,x_sampling=fD_sampling,y_sampling=stau_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([fD_min,fD_max])
    ax.set_ylim([stau_min,stau_max])
    
    return im
    
def thfD_diagram(th,fD,thfD,ax,**kwargs):
    #parameters
    fD_sampling = kwargs.get("fD_sampling",1)
    th_sampling = kwargs.get("th_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"$\theta$-$f_D$-Diagram")
    ylabel = kwargs.get("ylabel",r"$f_D$ [mHz]")
    xlabel = kwargs.get("xlabel",r"$\theta$ [mas]")
    xscale = kwargs.get("xscale",mas)
    yscale = kwargs.get("yscale",mHz)
    fD_min = kwargs.get("fD_min",np.min(fD)/yscale)
    fD_max = kwargs.get("fD_max",np.max(fD)/yscale)
    th_min = kwargs.get("th_min",np.min(th)/xscale)
    th_max = kwargs.get("th_max",np.max(th)/xscale)
    log10 = kwargs.get("log10",True)
    
    #draw the plot
    im = colormesh(th/xscale,fD/yscale,thfD,ax,x_sampling=th_sampling,y_sampling=fD_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([th_min,th_max])
    ax.set_ylim([fD_min,fD_max])
    
    return im
    
def thth_diagram(th,thth,ax,**kwargs):
    #parameters
    th1_sampling = kwargs.get("th_sampling",1)
    th2_sampling = kwargs.get("th2_sampling",th1_sampling)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"$\theta$-$\theta$ Diagram")
    ylabel = kwargs.get("ylabel",r"$\theta$ [mas]")
    xlabel = kwargs.get("xlabel",r"$\theta$ [mas]")
    xscale = kwargs.get("xscale",mas)
    yscale = kwargs.get("yscale",mas)
    th_min = kwargs.get("th_min",np.min(th)/xscale)
    th_max = kwargs.get("th_max",np.max(th)/xscale)
    
    #draw the plot
    im = colormesh(th/xscale,th/yscale,thth,ax,x_sampling=th1_sampling,y_sampling=th2_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([th_min,th_max])
    ax.set_ylim([th_min,th_max])
    
    return im

def thth_arc(thx,thy,thth,ax,**kwargs):
    #parameters
    thx_sampling = kwargs.get("thx_sampling",1)
    thy_sampling = kwargs.get("thy_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"$\theta$-$\theta$ Diagram")
    ylabel = kwargs.get("ylabel",r"$\theta_x$ [mas]")
    xlabel = kwargs.get("xlabel",r"$\theta_y$ [mas]")
    xscale = kwargs.get("xscale",mas)
    yscale = kwargs.get("yscale",mas)
    thx_min = kwargs.get("thx_min",np.min(thx)/xscale)
    thx_max = kwargs.get("thx_max",np.max(thx)/xscale)
    thy_min = kwargs.get("thy_min",np.min(thy)/xscale)
    thy_max = kwargs.get("thy_max",np.max(thy)/xscale)
    
    #draw the plot
    im = colormesh(thx/xscale,thy/yscale,thth,ax,x_sampling=thx_sampling,y_sampling=thy_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([thx_min,thx_max])
    ax.set_ylim([thy_min,thy_max])
    
    return im
    
def SkyMap(th_par,th_ort,Map,ax,**kwargs):
    #parameters
    x_sampling = kwargs.get("x_sampling",1)
    y_sampling = kwargs.get("y_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Map on Sky")
    ylabel = kwargs.get("xlabel",r"$\theta_\perp$ [mas]")
    xlabel = kwargs.get("xlabel",r"$\theta_\parallel$ [mas]")
    xscale = kwargs.get("xscale",mas)
    yscale = kwargs.get("yscale",mas)
    th_par_min = kwargs.get("th_par_min",np.min(th_par)/xscale)
    th_par_max = kwargs.get("th_par_max",np.max(th_par)/xscale)
    th_ort_min = kwargs.get("th_ort_min",np.min(th_ort)/xscale)
    th_ort_max = kwargs.get("th_ort_max",np.max(th_ort)/xscale)
    
    #draw the plot
    im = colormesh(th_par/xscale,th_ort/yscale,Map,ax,x_sampling=x_sampling,y_sampling=y_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([th_par_min,th_par_max])
    ax.set_ylim([th_ort_min,th_ort_max])
    
    return im
    
def theta_freq(th,nu,thnu,ax,**kwargs):
    #parameters
    th_sampling = kwargs.get("th_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Eigenvector")
    xlabel = kwargs.get("xlabel",r"$\theta$ [mas]")
    ylabel = kwargs.get("ylabel",r"$\nu$ [MHz]")
    xscale = kwargs.get("xscale",mas)
    yscale = kwargs.get("yscale",MHz)
    th_min = kwargs.get("th_min",np.min(th)/xscale)
    th_max = kwargs.get("th_max",np.max(th)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/yscale)
    log10 = kwargs.get("log10",False)
    
    #draw the plot
    im = colormesh(th/xscale,nu/yscale,thnu,ax,x_sampling=th_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([th_min,th_max])
    ax.set_ylim([nu_min,nu_max])
    
    return im
    
def freq_theta(nu,th,thnu,ax,**kwargs):
    #parameters
    th_sampling = kwargs.get("th_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Eigenvector")
    xlabel = kwargs.get("xlabel",r"$\nu$ [MHz]")
    ylabel = kwargs.get("ylabel",r"$\theta$ [mas]")
    xscale = kwargs.get("xscale",MHz)
    yscale = kwargs.get("yscale",mas)
    th_min = kwargs.get("th_min",np.min(th)/xscale)
    th_max = kwargs.get("th_max",np.max(th)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/yscale)
    log10 = kwargs.get("log10",False)
    
    #draw the plot
    im = colormesh(nu/xscale,th/yscale,thnu,ax,y_sampling=th_sampling,x_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim([th_min,th_max])
    ax.set_xlim([nu_min,nu_max])
    
    return im
    
def time_theta(t,th,tth,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    th_sampling = kwargs.get("th_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Eigenvector")
    xlabel = kwargs.get("xlabel",r"$t$ [min]")
    ylabel = kwargs.get("ylabel",r"$\theta$ [mas]")
    xscale = kwargs.get("xscale",minute)
    yscale = kwargs.get("yscale",mas)
    t_min = kwargs.get("t_min",np.min(t)/xscale)
    t_max = kwargs.get("t_max",np.max(t)/xscale)
    th_min = kwargs.get("th_min",np.min(th)/yscale)
    th_max = kwargs.get("th_max",np.max(th)/yscale)
    log10 = kwargs.get("log10",False)
    
    #draw the plot
    im = colormesh(t/xscale,th/yscale,tth,ax,x_sampling=t_sampling,y_sampling=th_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([t_min,t_max])
    ax.set_ylim([th_min,th_max])
    
    return im
    
def theta_time(th,t,tht,ax,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    th_sampling = kwargs.get("th_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Eigenvector")
    xlabel = kwargs.get("xlabel",r"$\theta$ [mas]")
    ylabel = kwargs.get("ylabel",r"$t$ [min]")
    xscale = kwargs.get("xscale",minute)
    yscale = kwargs.get("yscale",mas)
    t_min = kwargs.get("t_min",np.min(t)/yscale)
    t_max = kwargs.get("t_max",np.max(t)/yscale)
    th_min = kwargs.get("th_min",np.min(th)/xscale)
    th_max = kwargs.get("th_max",np.max(th)/xscale)
    log10 = kwargs.get("log10",False)
    
    #draw the plot
    im = colormesh(th/xscale,t/yscale,tht,ax,x_sampling=th_sampling,y_sampling=t_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=log10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([th_min,th_max])
    ax.set_ylim([t_min,t_max])
    
    return im

def scintscales(t_shift,nu_shift,ACF,ccorr_t,t_model_ACF,ccorr_nu,nu_model_ACF,figure,**kwargs):
    #parameters
    t_sampling = kwargs.get("t_sampling",1)
    nu_sampling = kwargs.get("nu_sampling",1)
    cmap = kwargs.get("cmap",'Greys')
    vmin = kwargs.get("vmin",np.mean(ACF) - 3*np.std(ACF))
    vmax = kwargs.get("vmax",np.max(ACF))
    title = kwargs.get("title",r"ACF")
    xlabel = kwargs.get("xlabel",r"$\Delta t$ [min]")
    ylabel = kwargs.get("ylabel",r"$\Delta\nu$ [MHz]")
    xscale = kwargs.get("xscale",minute)
    yscale = kwargs.get("yscale",MHz)
    t_min = kwargs.get("t_min",np.min(t_shift)/xscale)
    t_max = kwargs.get("t_max",np.max(t_shift)/xscale)
    nu_min = kwargs.get("nu_min",np.min(nu_shift)/yscale)
    nu_max = kwargs.get("nu_max",np.max(nu_shift)/yscale)
    model_color = kwargs.get("model_color",'tab:blue')

    ax1 = plt.subplot2grid((4, 4), (1, 0), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((4, 4), (1, 3), rowspan=3)
    ax3 = plt.subplot2grid((4, 4), (0, 0), colspan=3)
    
    im = colormesh(t_shift/xscale,nu_shift/yscale,ACF,ax1,x_sampling=t_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=False)
    ax2.plot( ccorr_nu, nu_shift/yscale, color='k')
    ax2.plot(nu_model_ACF, nu_shift/yscale, color=model_color, linestyle='--')
    ax3.plot( t_shift/xscale, ccorr_t, color='k')
    ax3.plot( t_shift/xscale, t_model_ACF, color=model_color, linestyle='--')
    
    
    figure.suptitle(title)
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax3.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax3.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    ax1.set_xlim([t_min,t_max])
    ax1.set_ylim([nu_min,nu_max])
    ax3.set_xlim([t_min,t_max])
    ax2.set_ylim([nu_min,nu_max])
    
def diagnostic(t,nu,DS,fD,tau,SS,zeta,zeta_err,zeta_scales,t_shift,nu_shift,ACF,ccorr_t,t_model_ACF,ccorr_nu,nu_model_ACF,figure,**kwargs):
    #parameters
    cmap = kwargs.get("cmap",'viridis')
    title = kwargs.get("title",r"unknown date, telescope, and pulsar")
    
    #DS parameters
    title_DS = kwargs.get("title_DS",r"Dynamic Spectrum")
    t_sampling = kwargs.get("d_sampling",1)
    nu_sampling = kwargs.get("d_sampling",1)
    vmin_DS = kwargs.get("vmin_DS",0.)
    vmax_DS = kwargs.get("vmax_DS",7.)
    tlabel = kwargs.get("tlabel",r"$ t$ [min]")
    nulabel = kwargs.get("nulabel",r"$\nu$ [MHz]")
    tscale = kwargs.get("tscale",minute)
    nuscale = kwargs.get("nuscale",MHz)
    t_min = kwargs.get("t_min",np.min(t)/tscale)
    t_max = kwargs.get("t_max",np.max(t)/tscale)
    nu_min = kwargs.get("nu_min",np.min(nu)/nuscale)
    nu_max = kwargs.get("nu_max",np.max(nu)/nuscale)
    
    nu0 = np.mean(nu)
    
    #ACF parameters
    title_ACF = kwargs.get("title_ACF","Autocorrelation")
    dt_sampling = kwargs.get("dt_sampling",1)
    dnu_sampling = kwargs.get("dnu_sampling",1)
    vmin_ACF = kwargs.get("vmin_ACF",np.mean(ACF) - 3*np.std(ACF))
    vmax_ACF = kwargs.get("vmax_ACF",np.mean(ACF) + 10*np.std(ACF))
    dtlabel = kwargs.get("dtlabel",r"$\Delta t$ [min]")
    dnulabel = kwargs.get("dnulabel",r"$\Delta\nu$ [MHz]")
    dtscale = kwargs.get("dtscale",minute)
    dnuscale = kwargs.get("dnuscale",MHz)
    dt_min = kwargs.get("dt_min",3./4.*np.min(t_shift)/dtscale)
    dt_max = kwargs.get("dt_max",3./4.*np.max(t_shift)/dtscale)
    dnu_min = kwargs.get("dnu_min",3./4.*np.min(nu_shift)/dnuscale)
    dnu_max = kwargs.get("dnu_max",3./4.*np.max(nu_shift)/dnuscale)
    model_color = kwargs.get("model_color",'tab:blue')
    
    #SS parameters
    title_SS = kwargs.get("title_SS",r"Secondary Spectrum (NuT)")
    fD_sampling = kwargs.get("fD_sampling",1)
    tau_sampling = kwargs.get("tau_sampling",1)
    vmin_SS = kwargs.get("vmin_SS",np.log10(0.9*np.median(SS)))
    vmax_SS = kwargs.get("vmax_SS",np.log10(0.9*np.max(SS)))
    fDlabel = kwargs.get("fDlabel",r"$f_D$ [mHz]")
    taulabel = kwargs.get("taulabel",r"$\tau$ [$\mu$s]")
    fDscale = kwargs.get("fDscale",mHz)
    tauscale = kwargs.get("tauscale",musec)
    fD_min = kwargs.get("fD_min",np.min(fD)/fDscale)
    fD_max = kwargs.get("fD_max",np.max(fD)/fDscale)
    tau_min = kwargs.get("tau_min",np.min(tau)/tauscale)
    tau_max = kwargs.get("tau_max",np.max(tau)/tauscale)
    arc_color = kwargs.get("arc_color",'white')
    
    ax1 = plt.subplot2grid((8, 8), (1, 4), colspan=3, rowspan=3)
    ax2 = plt.subplot2grid((8, 8), (1, 7), rowspan=3)
    ax3 = plt.subplot2grid((8, 8), (0, 4), colspan=3)
    ax4 = plt.subplot2grid((8, 8), (4, 4), colspan=4, rowspan=4)
    ax5 = plt.subplot2grid((8, 8), (0, 0), colspan=4, rowspan=8)
    
    colormesh(t_shift/dtscale,nu_shift/dnuscale,ACF,ax1,x_sampling=dt_sampling,y_sampling=dnu_sampling,cmap=cmap,vmin=vmin_ACF,vmax=vmax_ACF,log10=False)
    ax2.plot( ccorr_nu, nu_shift/dnuscale, color='k')
    ax2.plot(nu_model_ACF, nu_shift/dnuscale, color=model_color, linestyle='--')
    ax1.set_title(title_ACF)
    ax3.plot( t_shift/tscale, ccorr_t, color='k')
    ax3.plot( t_shift/tscale, t_model_ACF, color=model_color, linestyle='--')
    
    figure.suptitle(title)
    
    ax3.xaxis.set_label_position("top")
    ax3.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax3.set_yticks([])
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax3.set_xlabel(dtlabel)
    ax2.set_ylabel(dnulabel)
    ax1.set_xlim([dt_min,dt_max])
    ax1.set_ylim([dnu_min,dnu_max])
    ax3.set_xlim([dt_min,dt_max])
    ax2.set_ylim([dnu_min,dnu_max])
    
    colormesh(fD/fDscale,tau/tauscale,SS,ax5,x_sampling=fD_sampling,y_sampling=tau_sampling,cmap=cmap,vmin=vmin_SS,vmax=vmax_SS,log10=True)
    ax5.set_title(title_SS)
    ax5.set_xlabel(fDlabel)
    ax5.set_ylabel(taulabel)
    ax5.set_xlim([fD_min,fD_max])
    ax5.set_ylim([tau_min,tau_max])
    tau_plot = np.linspace(-0.1*tau_max*tauscale,-tau_max*tauscale,num=50,endpoint=True)
    eta_min = 1./(2.*nu0*(zeta-zeta_err))**2
    eta_max = 1./(2.*nu0*(zeta+zeta_err))**2
    fD_plot_min = np.sqrt(np.abs(tau_plot)/eta_min)
    fD_plot_max = np.sqrt(np.abs(tau_plot)/eta_max)
    ax5.fill_betweenx(tau_plot/tauscale,fD_plot_min/fDscale,fD_plot_max/fDscale,color=arc_color,alpha=0.5)
    ax5.fill_betweenx(tau_plot/tauscale,-fD_plot_min/fDscale,-fD_plot_max/fDscale,color=arc_color,alpha=0.5)
    eta_scales = 1./(2.*nu0*zeta_scales)**2
    fD_plot = np.sqrt(np.abs(tau_plot)/eta_scales)
    ax5.plot(fD_plot/fDscale,tau_plot/tauscale,color=arc_color,linestyle="--")
    ax5.plot(-fD_plot/fDscale,tau_plot/tauscale,color=arc_color,linestyle="--")
    #print(tau_plot[0]/tauscale,fD_plot_min[0]/fDscale)
    print(tau_plot[0]/tauscale,fD_plot[0]/fDscale)
    
    colormesh(t/tscale,nu/nuscale,DS,ax4,x_sampling=t_sampling,y_sampling=nu_sampling,cmap=cmap,vmin=vmin_DS,vmax=vmax_DS)
    ax4.set_title(title_DS)
    ax4.set_xlabel(tlabel)
    ax4.set_ylabel(nulabel)
    ax4.set_xlim([t_min,t_max])
    ax4.set_ylim([nu_min,nu_max])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

def refraction_spectrum(fD,fnum2,SS,ax,**kwargs):
    #parameters
    fD_sampling = kwargs.get("fD_sampling",1)
    fnum2_sampling = kwargs.get("fnum2_sampling",1)
    cmap = kwargs.get("cmap",'viridis')
    vmin = kwargs.get("vmin",None)
    vmax = kwargs.get("vmax",None)
    title = kwargs.get("title",r"Refraction Spectrum")
    xlabel = kwargs.get("xlabel",r"$f_D$ [mHz]")
    ylabel = kwargs.get("ylabel",r"$f_{1/\nu^2}$ [MHz$^2$]")
    xscale = kwargs.get("xscale",mHz)
    yscale = kwargs.get("yscale",MHz**2)
    fD_min = kwargs.get("fD_min",np.min(fD)/xscale)
    fD_max = kwargs.get("fD_max",np.max(fD)/xscale)
    fnum2_min = kwargs.get("fnum2_min",np.min(fnum2)/yscale)
    fnum2_max = kwargs.get("fnum2_max",np.max(fnum2)/yscale)
        
    #draw the plot
    im = colormesh(fD/xscale,fnum2/yscale,SS,ax,x_sampling=fD_sampling,y_sampling=fnum2_sampling,cmap=cmap,vmin=vmin,vmax=vmax,log10=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim([fD_min,fD_max])
    ax.set_ylim([fnum2_min,fnum2_max])
    
    return im

def convert_png_to_gif(input_folder,output_file,duration,loop):
    images =[]
    png_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".png")]
    png_files.sort()
    for png_file in png_files:
        png_path = os.path.join(input_folder, png_file)
        img = Image.open(png_path)
        images.append(img)
    images[0].save(os.path.join(input_folder,output_file), save_all=True, append_images=images[1:],duration=duration,loop=loop)
