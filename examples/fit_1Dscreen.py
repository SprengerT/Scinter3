 
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.optimize import minimize
from astropy.coordinates import EarthLocation, Angle
import emcee
import corner
import progressbar

# import scinter
sys.path.insert(0,'/path/to/Scinter3')
import scinter_computation
import scinter_import
import scinter_measurement
import scinter_data
import scinter_plot

plt.style.use('fast')

# some useful constants in SI units
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
kHz = 1.0e+3
MHz = 1.0e+6
GHz = 1.0e+9
mus = 1.0e-6
sqrtmus = np.sqrt(mus)
kms = 1000.

"""
This script makes use of ctypes which do not properly work in ipython and on MacOS.
For the purposes of this tutorial, individual steps are separated by 'if 0:' clauses
that can be turned on and off individually.
We start from an already created storage of observations. A tutrial to create such a
storage can be found in the exmple Analyze_Observation.ipynb .
"""

path_results = "/path/chosen/by/you/for/processed/data/and/results"

storage = scinter_measurement.storage(path_results)
# choose the list of observations, e.g. "all"
list_tutorial = "tutorial"

obs_list = storage.obs_lists[list_tutorial]

if 0:
    """
    Plot all secondary spectra in one grid to study their evolution.
    """
    
    vmin = 4.5
    vmax = 8.3
    fD_sampling = 1
    tau_sampling = 6
    fD_max = 19.
    tau_max = 12
    cmap = "viridis"
    
    N_obs = len(obs_list)
    N_x = int(np.sqrt(N_obs))
    N_y = int(np.ceil(N_obs/N_x))
    print("N_obs = {0} => {1}x{2}".format(N_obs,N_x,N_y))
    N_empty = N_x*N_y-N_obs
    assert N_empty>=0
    i_x_end = N_x - N_empty
    
    figure = scinter_plot.draw_canvas(plot_width = 3000,plot_height = int(3000*N_y/N_x), plot_bottom = 0.05, plot_left = 0.05, plot_wspace = 0.07, plot_hspace = 0.07, textsize=36, labelsize=36)
    
    bar = progressbar.ProgressBar(maxval=N_obs, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for i,obsname in enumerate(obs_list):
        bar.update(i)
        obs = scinter_measurement.measurement(storage.data_path,obsname)
        DS = scinter_data.intensity(obs.data_path)
        FFT = scinter_data.SecSpec_FFT(DS,data_path=obs.data_path,overwrite=False)
        
        i_y = int(i/N_x)
        i_x = i%N_x
        ax = figure.add_subplot(N_y,N_x,i+1)
        
        plot = scinter_plot.secondary_spectrum(FFT.fD,FFT.tau,FFT.SS,ax,fD_sampling=fD_sampling,tau_sampling=tau_sampling,fD_min=-fD_max,fD_max=fD_max,tau_min=0.,tau_max=tau_max,vmin=vmin,vmax=vmax,title="",cmap=cmap)
        if i_x!=0:
            ax.set_ylabel("")
            ax.set_yticks([])
        if (i_y!=N_y-1 and i_x<i_x_end) or (i_y!=N_y-2 and i_x>=i_x_end):
            ax.set_xlabel("")
            ax.set_xticks([])
            
        ax.text(0.38,0.86,"{0:.0f}".format(DS.mjd0),color="white",fontsize=36,transform = ax.transAxes)
    bar.finish()
        
    figure.savefig(os.path.join(storage.data_path,"SecSpec_collection_{0}.png".format(list_tutorial)))
    plt.close()

if 0:
    """
    Measure the arc curvatures by manually fitting lines to sqrt(tau)-FD diagrams that were scaled to 1.35 GHz.
    They are scaled to the central frequency of each observation.
    The arc curvature is expressed as zeta = 1/(2 nu sqrt(eta)).
    """
    for obsname in obs_list:
        obs = scinter_measurement.measurement(storage.data_path,obsname)
        DS = scinter_data.intensity(obs.data_path)
        
        nu0 = DS.nu0
        SecSpec = scinter_data.SecSpec_NuT(DS,file_name="SecSpec_NuT_1350.npz",data_path=obs.data_path,nu0=nu0,overwrite=False)
        staufD = scinter_data.staufD(SecSpec,data_path=obs.data_path,N_stau=200,remove_noise=False,overwrite=False)
        # The parameters of the fitting method need to be manually optimized.
        zeta, zeta_err = staufD.get_zeta(vmin=2.,vmax=None,nu0_data=nu0,zeta_max=0.5e-8,zeta_init=0.0,xmin=-20.,xmax=20.)
        
        obs.enter_result("zeta_manual",zeta,dtype=float)
        obs.enter_result("zeta_err_manual",zeta_err,dtype=float)
        plt.close()
        
        print("{0}: zeta={1} +- {2}".format(obsname,obs.results["zeta_manual"],obs.results["zeta_err_manual"]))
        
if 0:
    """
    Plot the zeta values that you measured. To successfully fit a 1D screen model,
    visible annual/orbital variation is necessary.
    """
    arr_mjd = storage.get_array(list_tutorial,"mjd0")
    arr_zeta = storage.get_array(list_tutorial,"zeta_manual")
    arr_zeta_err = storage.get_array(list_tutorial,"zeta_err_manual")
    
    figure = scinter_plot.draw_canvas(plot_width = 2000,plot_height = 900, plot_bottom = 0.07, plot_left = 0.12, plot_top = 0.97, plot_right=0.95, plot_wspace = 0.1, plot_hspace = 0.1, textsize=18, labelsize=18)
    ax = figure.add_subplot(1,1,1)
    ax.errorbar(arr_mjd,arr_zeta/sqrtmus*hour,xerr=0.,yerr=arr_zeta_err/sqrtmus*hour,linestyle='',marker='o',markersize=5,color='black',label="tutorial data")
    
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$\zeta = \partial_t \sqrt{\tau}$   [$\sqrt{\mu s}$/h]")
    ax.legend()
    plt.grid(alpha=0.5,linestyle='--')
    figure.savefig(os.path.join(storage.data_path,"arc_curvatures_{0}.png".format(list_tutorial)))
    plt.close()
    
if 0:
    """
    Fit the model via minimization and via mcmc.
    Here, it is assumed that the source's velocity is constant.
    Otherwise, orbital variations need to be modeled.
    """
    arr_mjd = storage.get_array(list_tutorial,"mjd0")
    arr_zeta = storage.get_array(list_tutorial,"zeta_manual")
    arr_zeta_err = storage.get_array(list_tutorial,"zeta_err_manual")
    
    # Here, the pulsar B1508+55 and the Effelsberg telescope have been chosen as examples
    # The telcoords keyword could be omitted here because it is only needed for 
    # interferometric observables.
    telescope = EarthLocation.from_geodetic(Angle('6°52′58″'),Angle('50°31′29″'),height=319.)
    screen = scinter_computation.Evolution_One1DScreen(arr_mjd,'PSRB1508+55',telcoords=telescope)
    
    
    def log_likelihood(theta):
        a_x,D_x,V_x,D_s,PMRA,PMDEC = theta
        
        #compute observables from screen model
        model = screen.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,a_x=a_x,V_x=V_x)
        
        #compute likelihood
        likelihood = 0.
        likelihood += np.sum((arr_zeta-model['zeta1_x'])**2/arr_zeta_err**2)
        
        return -0.5*likelihood
    
    def get_chi2(theta,dof):
        a_x,D_x,V_x,D_s,PMRA,PMDEC = theta
        
        #compute observables from screen model
        model = screen.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,a_x=a_x,V_x=V_x)
        
        #compute likelihood
        chi2 = 0.
        count = 0
        chi2 += np.sum((arr_zeta-model['zeta1_x'])**2/arr_zeta_err**2)
        count += len(arr_zeta)
        
        return chi2/(count-dof)
    
    def log_prior(theta):
        a_x,D_x,V_x,D_s,PMRA,PMDEC = theta
        
        #apply prior
        prior = 0.
        if not (-180. < a_x < 180. and 0. < D_x < D_s):
            return -np.inf
        #Use results from Chatterjee et al. 2009:
        d_psr_mean = 2100.
        d_psr_std = 140.
        PMRA_mean = -73.64
        PMRA_std = 0.05
        PMDEC_mean = -62.65
        PMDEC_std = 0.09
        prior += (D_s-d_psr_mean)**2/d_psr_std**2 + (PMRA-PMRA_mean)**2/PMRA_std**2 + (PMDEC-PMDEC_mean)**2/PMDEC_std**2
        
        #Use ISM velocity prior by looking at typical solar neighborhood GAIA star velocities
        V_ISM_mean = 0.
        V_ISM_std = 100.
        prior += (V_x-V_ISM_mean)**2/V_ISM_std**2 
        
        return -0.5*prior
    
    def log_probability(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lk = lp + log_likelihood(theta)
        return lk
    
    # set up initial values
    D_s = 2100.     #pc
    PMRA = -73.64   #mas/year
    PMDEC = -62.65  #mas/year
    D_x = 120.     #pc
    a_x = 35.     #degrees (from ra axis)
    V_x = 0.       #km/s

    initial = np.array([a_x,D_x,V_x,D_s,PMRA,PMDEC])

    dof = 3
    soln = minimize(get_chi2, initial, args=(dof),bounds=[(-180.,180.),(0.,None),(None,None),(0.,None),(None,None),(None,None)])
    print(soln)
    initial = soln.x
    chi2 = get_chi2(soln.x,dof)
    print("chi2/ndf = {0}".format(chi2))
    
    mcmc_path = os.path.join(storage.data_path,"mcmc_chains")
    if not os.path.exists(mcmc_path):
        os.mkdir(mcmc_path)
    filename = os.path.join(mcmc_path,"chains.h5")

    # perform the MCMC

    nwalkers = 24
    ndim = len(initial)
    steps = 10000
    pos = initial[None,:] + 1. * np.random.randn(nwalkers, ndim)

    if not os.path.exists(filename):
        backend = emcee.backends.HDFBackend(filename)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend) #, args=(t, eta, err)
        sampler.run_mcmc(pos, steps, progress=True)
    else:
        backend = emcee.backends.HDFBackend(filename)
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, backend=backend) #, args=(t, eta, err)
        sampler.run_mcmc(None, steps, progress=True)
        
    # plot the triangle plot
    reader = emcee.backends.HDFBackend(filename)
    flat_samples = reader.get_chain(discard=0, flat=True)
    N_steps = flat_samples.shape[0]/nwalkers
    burnin = int(N_steps/2.)
    print("burnin: {0}".format(burnin))
    flat_samples = reader.get_chain(discard=burnin, flat=True)
    results_mean = np.mean(flat_samples,axis=0)
    results_std = np.std(flat_samples,axis=0)

    #write results to file
    with open(os.path.join(mcmc_path,"results.txt"),'w') as writefile:
        writefile.write("a_x\tD_x\tV_x\tD_s\tPMRA\tPMDEC\n")
        writefile.write("bestfit, chi2: {0}\n".format(chi2))
        writefile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*soln.x))
        writefile.write("mcmc mean\n")
        writefile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*results_mean))
        writefile.write("mcmc std\n")
        writefile.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(*results_std))

    figure = scinter_plot.draw_canvas(plot_width = 2000,plot_height = 2000, plot_bottom = 0.12, plot_left = 0.12, plot_top = 0.97, plot_right=0.95, plot_wspace = 0.1, plot_hspace = 0.1, textsize=18, labelsize=18)
    labels = [r'$\alpha_{x}$ [°]',r'$D_{x}$ [pc]',r'$V_{x}$ [km/s]',r'$D_{s}$ [pc]',r'PMRA [mas/yr]',r'PMDEC [mas/yr]']
    corner.corner(flat_samples,labels=labels,fig=figure)
    figure.savefig(os.path.join(mcmc_path,"triangle.png"))
    plt.close()
    
    
    figure = scinter_plot.draw_canvas(plot_width = 2000,plot_height = 900, plot_bottom = 0.07, plot_left = 0.12, plot_top = 0.97, plot_right=0.95, plot_wspace = 0.1, plot_hspace = 0.1, textsize=18, labelsize=18)
    ax = figure.add_subplot(1,1,1)
    
    #plot MCMC results
    mjd_plot = np.linspace(np.min(arr_mjd),np.max(arr_mjd),num=200)
    screen_plot = scinter_computation.Evolution_One1DScreen(mjd_plot,'PSRB1508+55',telcoords=telescope)
    N_sample = 1000
    inds = np.random.randint(len(flat_samples), size=N_sample)
    for i,ind in enumerate(inds):
        a_x,D_x,V_x,D_s,PMRA,PMDEC = flat_samples[ind]
        
        #compute observables from screen model
        model = screen_plot.compute(D_s=D_s,PMRA=PMRA,PMDEC=PMDEC,D_x=D_x,a_x=a_x,V_x=V_x)
        
        ax.plot(mjd_plot,model['zeta1_x']/sqrtmus*hour,linestyle='-',markersize=0,color='black',alpha=0.01)
    
    ax.errorbar(arr_mjd,arr_zeta/sqrtmus*hour,xerr=0.,yerr=arr_zeta_err/sqrtmus*hour,linestyle='',marker='o',markersize=5,color='black',label="tutorial data")
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$\zeta = \partial_t \sqrt{\tau}$   [$\sqrt{\mu s}$/h]")
    ax.legend()
    plt.grid(alpha=0.5,linestyle='--')
    figure.savefig(os.path.join(mcmc_path,"arc_curvatures_fit_{0}.png".format(list_tutorial)))
    plt.close()
    
if 0:
    if 0:
        # compute scintillation scales for all observations in the list
        # If you followed Analyze_Observation.ipynb, jump to the last step instead.
        for obsname in obs_list:
            obs = scinter_measurement.measurement(storage.data_path,obsname)
            DS = scinter_data.intensity(obs.data_path)
            ACF = scinter_data.ACF_DS(DS,data_path=obs.data_path,overwrite=False)
            nuscint,tscint,modindex,model_ACF,ccorr_t,ccorr_nu,t_model_ACF,nu_model_ACF = ACF.fit2D()
            obs.enter_result('tscint',tscint)
            obs.enter_result('nuscint',nuscint)
            
            # plot diagnostic plot
            figure = scinter_plot.draw_canvas(plot_width = 800,plot_height = 600, plot_bottom = 0.1, plot_left = 0.05, plot_top=0.9,plot_right=0.9, plot_wspace = 0.1, plot_hspace = 0.1, textsize=16, labelsize=12)
            scinter_plot.scintscales(ACF.t_shift,ACF.nu_shift,ACF.ACF,ccorr_t,t_model_ACF,ccorr_nu,nu_model_ACF,figure,nu_sampling=1,t_min=-5*tscint/minute,t_max=5*tscint/minute,nu_min=-5*nuscint/MHz,nu_max=5*nuscint/MHz,cmap="viridis",corr_max=1.2*modindex**2)
            plt.text(0.8, 0.75, "source name\n"+"{:.0f}".format(DS.nu0/MHz)+" MHz\n$t_{\\mathrm{ISS}}="+"{:.1f}".format(tscint)+"$ s\n$\\nu_{\\mathrm{ISS}}="+"{:.3f}".format(nuscint/kHz)+"$ kHz", transform=plt.gcf().transFigure)
            figure.savefig(os.path.join(obs.obs_path,"scintscales.png"))
        
        # plot the variation of the scales over time
    
        arr_mjd = storage.get_array(list_tutorial,"mjd0")
        arr_nuscint = storage.get_array(list_tutorial,"nuscint")
        arr_tscint = storage.get_array(list_tutorial,"tscint")
        
        figure = scinter_plot.draw_canvas(plot_width = 800,plot_height = 800, plot_bottom = 0.1, plot_left = 0.1, plot_wspace = 0.1, plot_hspace = 0.1, textsize=16, labelsize=12)
        ax1 = figure.add_subplot(2,1,1)
        ax2 = figure.add_subplot(2,1,2)
        ax1.plot(arr_mjd,arr_nuscint/kHz,marker='o',linestyle='')
        ax2.plot(arr_mjd,arr_tscint,marker='o',linestyle='')
        ax2.set_xlabel("MJD")
        ax1.set_ylabel(r"$\nu_{\rm ISS}$ [kHz]")
        ax2.set_ylabel(r"$t_{\rm ISS}$ [s]")
        figure.savefig(os.path.join(storage.data_path,"scintscales_{0}.png".format(list_tutorial)))
    
    # translate the scintillation scales to something proportional to zeta and compare
    # to the zeta obtained from arc curvatures

    arr_mjd = storage.get_array(list_tutorial,"mjd0")
    arr_nu0 = storage.get_array(list_tutorial,"nu0")
    arr_nuscint = storage.get_array(list_tutorial,"nuscint")
    arr_tscint = storage.get_array(list_tutorial,"tscint")
    arr_zeta = storage.get_array(list_tutorial,"zeta_manual")
    arr_zeta_err = storage.get_array(list_tutorial,"zeta_err_manual")
    
    eta_iss = 2*np.pi*arr_tscint**2/arr_nuscint
    zeta_iss = 1./(2.*arr_nu0*np.sqrt(eta_iss))
    
    figure = scinter_plot.draw_canvas(plot_width = 2000,plot_height = 900, plot_bottom = 0.07, plot_left = 0.12, plot_top = 0.97, plot_right=0.95, plot_wspace = 0.1, plot_hspace = 0.1, textsize=18, labelsize=18)
    ax = figure.add_subplot(1,1,1)
    ax.errorbar(arr_mjd,arr_zeta/sqrtmus*hour,xerr=0.,yerr=arr_zeta_err/sqrtmus*hour,linestyle='',marker='o',markersize=5,color='blue',label="arc curvature")
    ax.errorbar(arr_mjd,zeta_iss/sqrtmus*hour,xerr=0.,yerr=0.,linestyle='',marker='o',markersize=5,color='red',label="scintillation scales")
    ax.set_xlabel("MJD")
    ax.set_ylabel(r"$\zeta = \partial_t \sqrt{\tau}$   [$\sqrt{\mu s}$/h]")
    ax.legend()
    plt.grid(alpha=0.5,linestyle='--')
    figure.savefig(os.path.join(storage.data_path,"zeta_methods_{0}.png".format(list_tutorial)))
    plt.close()
    
