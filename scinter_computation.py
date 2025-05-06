# -*- coding: utf-8 -*-

import numpy as np
from numpy import newaxis as na
import os
from numpy.ctypeslib import ndpointer
import ctypes
import progressbar
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, get_body_barycentric, get_body, get_body_barycentric_posvel, SkyCoord, EarthLocation, Angle
from astropy import units as u
from scipy.interpolate import interp1d

#constants
au = 149597870700. #m
pc = 648000./np.pi*au #m
day = 24*3600
year = 365.2425*day
degrees = np.pi/180.
mas = degrees/1000./3600.
v_c = 299792458. #m/s
MHz = 1.0e+6
GHz = 1.0e+6
mHz = 1.0e-3
musec = 1.0e-6
mus = 1.0e-6
e = 1.602176634e-19 #C
me = 9.1093837015e-31 #kg
eps0 = 8.8541878128e-12 #SI
pc_per_cm3 = pc/0.01**3
hour = 3600.
minute = 60.
kms = 1000.
f_DM = -e**2/(4.*np.pi*eps0*me*v_c)*pc_per_cm3

try:
    #import C++ library
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_c = os.path.join(os.path.join(dir_path,"libcpp"),"lib_scinter.so")
    lib = ctypes.CDLL(file_c)

    #load C++ library for fast SP trafo
    lib.SumStatPoints.argtypes = [
        ctypes.c_int,   # N_nu
        ctypes.c_int,   # N_th
        ctypes.c_int,   # N_t
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # ph [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # psi [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # wt [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ]

    #load C++ library for fast SP trafo
    lib.SumStatPoints_GlobTVar.argtypes = [
        ctypes.c_int,   # N_nu
        ctypes.c_int,   # N_th
        ctypes.c_int,   # N_t
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # ph [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # psi [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # wt [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # TVar [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ]

    #load C++ library for fast SP trafo
    lib.SumStatPoints_SingleTVar.argtypes = [
        ctypes.c_int,   # N_nu
        ctypes.c_int,   # N_th
        ctypes.c_int,   # N_t
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # ph [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # psi [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # wt [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # TVar [N_th,N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ]

    #load C++ library for fast SP trafo
    lib.SumStatPoints_TDrift.argtypes = [
        ctypes.c_int,   # N_nu
        ctypes.c_int,   # N_th
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_D
        ctypes.c_double,   # slope
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # ph [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # psi [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # wt [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # psi_Drift [N_D]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # TDrift [N_D]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
    ]
        
    #load C++ library for fast SP computation of a system of two 1D screens
    lib.SumStatPoints_2scr.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ctypes.c_int,   # N_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_y
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ctypes.c_double,   # V_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # a_x
        ctypes.c_double,   # a_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
    ]
    lib.SumStatPoints_2scr_CP.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ctypes.c_int,   # N_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_y
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ctypes.c_double,   # V_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # a_x
        ctypes.c_double,   # a_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
        ctypes.c_double,   # mu_CPx
        ctypes.c_double,   # mu_CPy
        ctypes.c_double,   # p_ra
        ctypes.c_double,   # p_dec
        ctypes.c_double,   # s_ra
        ctypes.c_double,   # s_dec
    ]

    lib.SumStatPoints_2scr_atTel_CP.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ctypes.c_int,   # N_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_y
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ctypes.c_double,   # V_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # a_x
        ctypes.c_double,   # a_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
        ctypes.c_double,   # mu_CPx
        ctypes.c_double,   # mu_CPy
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),   # p_ra
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),   # p_dec
        ctypes.c_double,   # s_ra
        ctypes.c_double,   # s_dec
    ]

    lib.SumStatPoints_1scr_atTel_CP.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # a_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
    ]

    lib.SumStatPoints_1scr_PS_CP.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # s_par
        ctypes.c_double,   # a_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
        ctypes.c_double,   # mu_CPx
    ]

    lib.SumStatPoints_1scr_PS_CP_2D.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x_ra [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x_dec [N_x]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x_ra
        ctypes.c_double,   # V_x_dec
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # s_ra
        ctypes.c_double,   # s_dec
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_x [N_x]
        ctypes.c_double,   # mu_CPx
    ]

    # lib.SumStatPoints_CPx.argtypes = [
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        # ctypes.c_int,   # N_t
        # ctypes.c_int,   # N_nu
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        # ctypes.c_int,   # N_y
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        # ctypes.c_double,   # D_x
        # ctypes.c_double,   # D_y
        # ctypes.c_double,   # D_s
        # ctypes.c_double,   # V_x
        # ctypes.c_double,   # V_y
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        # ctypes.c_double,   # V_s_ra
        # ctypes.c_double,   # V_s_dec
        # ctypes.c_double,   # a_x
        # ctypes.c_double,   # a_y
        # ctypes.c_double,  # mu_x
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
    # ]

    # #load C++ library for fast SP computation of the E-field at the 2nd of two 1D screens
    # lib.SumStatPoints_at_2ndSCR.argtypes = [
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu*N_x]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu*N_x]
        # ctypes.c_int,   # N_t
        # ctypes.c_int,   # N_nu
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        # ctypes.c_int,   # N_x
        # ctypes.c_int,   # N_y
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # pos_x [N_x]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        # ctypes.c_double,   # D_x
        # ctypes.c_double,   # D_y
        # ctypes.c_double,   # D_s
        # ctypes.c_double,   # V_x
        # ctypes.c_double,   # V_y
        # ctypes.c_double,   # V_s
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        # ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
    # ]

    #load C++ library for fast SP computation of the E-field at the 2nd of two 1D screens
    lib.SumStatPoints_at_2ndSCR.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ctypes.c_int,   # N_x
        ctypes.c_int,   # N_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # im_y [N_y]
        ctypes.c_double,   # D_x
        ctypes.c_double,   # D_y
        ctypes.c_double,   # D_s
        ctypes.c_double,   # V_x
        ctypes.c_double,   # V_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_ra [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # V_p_dec [N_t]
        ctypes.c_double,   # V_s_ra
        ctypes.c_double,   # V_s_dec
        ctypes.c_double,   # a_x
        ctypes.c_double,   # a_y
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # mu_y [N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_y [N_y]
    ]

    #load C++ library for fast integration of 1D DM distribution for a single pixel
    lib.IntScreenSimple.argtypes = [
        ctypes.c_int,   # N_th
        ctypes.c_double,   # t
        ctypes.c_double,   # nu
        ctypes.c_double,   # V
        ctypes.c_double,   # thF
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # th [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # DM [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E
    ]

    #load C++ library for fast integration of 1D DM distribution for a line in time
    lib.DMI_stripe.argtypes = [
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_th
        ctypes.c_double,   # v_nu
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # thetas [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_disp [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_delay [N_th]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # phi_Doppler [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_im [N_t]
    ]

    #lib.DM2D_sinc(E_real,E_imag,N_t,N_nu,N_x,N_y,zeta_x,zeta_y,t,nu,self.stau_x,self.stau_y,self.M,self.M_dx,self.M_dy)
    lib.DM2D_sinc.argtypes = [
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_real [N_t*N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # E_imag [N_t*N_nu]
        ctypes.c_int,   # N_t
        ctypes.c_int,   # N_nu
        ctypes.c_int,   # N_x
        ctypes.c_int,   # N_y
        ctypes.c_double,   # zeta_x
        ctypes.c_double,   # zeta_x
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # t [N_t]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # nu [N_nu]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # stau_x [N_x]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # stau_y [N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # M [N_x*N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # M_dx [N_x*N_y]
        ndpointer(dtype=np.float64, flags='CONTIGUOUS', ndim=1),  # M_dy [N_x*N_y]
    ]
except:
    print("C++ part not working. Recompiling lib_scinter.cpp on your OS might help. Be aware that support for MacOS is missing. Scinter will continue without this functionality which might cause errors.")

def IntScreenSimple(N_th,t,nu,V,thF,th,DM):
    E = np.array([0.,0.])
    lib.IntScreenSimple(N_th,t,nu,V,thF,th,DM,E)
    E = E[0] + 1.0j*E[1]
    return E

class Screen1D:
    def __init__(self,pulsar,mjds,telescopes):
        self.N_data = len(mjds)
        self.mjds = mjds
    
        #get pulsar position
        psr = SkyCoord.from_name(pulsar)
        self.rarad = psr.ra.value * np.pi/180
        self.decrad = psr.dec.value * np.pi/180
        
        #get telescope position
        dict_tels = {}
        dict_tels.update({"Effelsberg":EarthLocation.from_geodetic(Angle('50°31′29″'),Angle('6°52′58″'),height=319.)})
        dict_tels.update({"LOFAR":EarthLocation.from_geodetic(Angle('52°54′19″'),Angle('6°52′5″'),height=23.)})
        if type(telescopes)==str:
            telescope = telescopes
            telescopes = [telescope for i in range(len(mjds))]
        locs = []
        for i,tel in enumerate(telescopes):
            locs.append(dict_tels[tel])
    
        #get telescope velocity
        self.vtel_ra = np.zeros_like(mjds)
        self.vtel_dec = np.zeros_like(mjds)
        for i,mjd in enumerate(mjds):
            time = Time(mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            pos_xyz, vel_xyz = locs[i].get_gcrs_posvel(time)
            vx += vel_xyz.x.value
            vy += vel_xyz.y.value
            vz += vel_xyz.z.value
            self.vtel_ra[i] = - vx * np.sin(self.rarad) + vy * np.cos(self.rarad)
            self.vtel_dec[i] = - vx * np.sin(self.decrad) * np.cos(self.rarad) - vy * np.sin(self.decrad) * np.sin(self.rarad) + vz * np.cos(self.decrad)
            
    def set_geometry(self,**kwargs):
        #load geometry parameters
        self.d_psr = kwargs.get("d_psr",10.)*pc
        self.d_scr = kwargs.get("d_scr",1.)*pc
        self.PMRA = kwargs.get("PMRA",0.)*mas/year
        self.PMDEC = kwargs.get("PMDEC",0.)*mas/year
        self.v_ISM_par = kwargs.get("v_ISM_par",0.)*kms
        self.angle = kwargs.get("angle",0.)*degrees
        
        #compute derived parameters
        self.d_scr_psr = self.d_psr-self.d_scr
        self.Deff = self.d_scr*self.d_psr/self.d_scr_psr
        
    def veff_par(self):
        """
        effective velocities along screen at times of observation taking earth rotation into account
        """
        #find total effective velocity in RA and DEC without ISM velocity
        pmra_v = self.PMRA*self.d_psr
        pmdec_v = self.PMDEC*self.d_psr
        veff_ra = self.vtel_ra + self.d_scr/self.d_scr_psr*pmra_v
        veff_dec = self.vtel_dec + self.d_scr/self.d_scr_psr*pmdec_v
        
        #compute effective velocity projected on screen
        angle_unitvec = np.array([np.cos(self.angle), np.sin(self.angle)])
        ve_vec = np.array([veff_ra, veff_dec])
        self.veff_par = np.zeros(self.N_data)
        norm = np.linalg.norm(angle_unitvec)
        for i in range(self.N_data):
            vv = ve_vec[:,i]
            self.veff_par[i] = np.dot(vv, angle_unitvec) / norm
            
        #add ISM velocity along screen
        self.veff_par += -self.d_psr/self.d_scr_psr*self.v_ISM_par
        
        return self.veff_par
        
    def veff_ort(self):
        """
        effective velocities perpendicular to screen at times of observation taking earth rotation into account
        """
        #find total effective velocity in RA and DEC without ISM velocity
        pmra_v = self.PMRA*self.d_psr
        pmdec_v = self.PMDEC*self.d_psr
        veff_ra = self.vtel_ra + self.d_scr/self.d_scr_psr*pmra_v
        veff_dec = self.vtel_dec + self.d_scr/self.d_scr_psr*pmdec_v
        
        #compute effective velocity projected on screen
        angle_unitvec = np.array([np.sin(self.angle), np.cos(self.angle)])
        ve_vec = np.array([veff_ra, veff_dec])
        self.veff_ort = np.zeros(self.N_data)
        norm = np.linalg.norm(angle_unitvec)
        for i in range(self.N_data):
            vv = ve_vec[:,i]
            self.veff_ort[i] = np.dot(vv, angle_unitvec) / norm
        
        return self.veff_ort
        
    def veff_par_at(self,mjd):
        """
        effective velocity along screen at given time, not accounting for earth rotation
        """
        #get earth velocity
        time = Time(mjd, format='mjd')
        pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
        vx = vel_xyz.x.to(u.m/u.s).value
        vy = vel_xyz.y.to(u.m/u.s).value
        vz = vel_xyz.z.to(u.m/u.s).value
        vtel_ra = - vx * np.sin(self.rarad) + vy * np.cos(self.rarad)
        vtel_dec = - vx * np.sin(self.decrad) * np.cos(self.rarad) - vy * np.sin(self.decrad) * np.sin(self.rarad) + vz * np.cos(self.decrad)
        
        #find total effective velocity in RA and DEC without ISM velocity
        pmra_v = self.PMRA*self.d_psr
        pmdec_v = self.PMDEC*self.d_psr
        veff_ra = vtel_ra + self.d_scr/self.d_scr_psr*pmra_v
        veff_dec = vtel_dec + self.d_scr/self.d_scr_psr*pmdec_v
        
        #compute effective velocity projected on screen
        angle_unitvec = np.array([np.cos(self.angle), np.sin(self.angle)])
        ve_vec = np.array([veff_ra, veff_dec])
        norm = np.linalg.norm(angle_unitvec)
        veff_par = np.dot(ve_vec, angle_unitvec) / norm
        veff_par +=  -self.d_psr/self.d_scr_psr*self.v_ISM_par
        
        return veff_par
        
    def veff_at(self,mjd):
        """
        effective velocity at given time, not accounting for earth rotation
        ra*dec
        """
        #get earth velocity
        time = Time(mjd, format='mjd')
        pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
        vx = vel_xyz.x.to(u.m/u.s).value
        vy = vel_xyz.y.to(u.m/u.s).value
        vz = vel_xyz.z.to(u.m/u.s).value
        vtel_ra = - vx * np.sin(self.rarad) + vy * np.cos(self.rarad)
        vtel_dec = - vx * np.sin(self.decrad) * np.cos(self.rarad) - vy * np.sin(self.decrad) * np.sin(self.rarad) + vz * np.cos(self.decrad)
        
        #find total effective velocity in RA and DEC without ISM velocity
        pmra_v = self.PMRA*self.d_psr
        pmdec_v = self.PMDEC*self.d_psr
        self.veff_ra = vtel_ra + self.d_scr/self.d_scr_psr*pmra_v
        self.veff_dec = vtel_dec + self.d_scr/self.d_scr_psr*pmdec_v
        
        #add ISM velocity along screen
        self.veff_ra += -self.d_psr/self.d_scr_psr*self.v_ISM_par*np.cos(self.angle)
        self.veff_dec += -self.d_psr/self.d_scr_psr*self.v_ISM_par*np.sin(self.angle)
        
        return self.veff_ra,self.veff_dec
        
    def eta_at(self,mjd,nu0):
        veff_par = self.veff_par_at(mjd)
        eta = v_c*self.Deff/(2.*nu0**2*veff_par**2)
        return eta
        
    def theta_los(self):
        """
        compute angular distance along screen travelled by the line of sight
        """
        i_t = 1
        integral = 0.
        int_step = 0.5
        timepoint = self.mjds[0]
        self.th_los = np.zeros(self.N_data)
        while i_t<self.N_data:
            dmjd = self.mjds[i_t]-timepoint
            if dmjd>int_step:
                integral += self.veff_par_at(timepoint)*int_step*day
                timepoint += int_step
            else:
                integral += self.veff_par_at(timepoint)*dmjd*day
                timepoint = self.mjds[i_t]
                self.th_los[i_t] = integral
                i_t += 1
        self.th_los = self.th_los / self.Deff
        
        return self.th_los
        
    def theta_los_2D(self):
        """
        compute angular distance travelled by the line of sight on 2D screen
        """
        i_t = 1
        integral_ra = 0.
        integral_dec = 0.
        int_step = 0.5
        timepoint = self.mjds[0]
        self.th_los_ra = np.zeros(self.N_data)
        self.th_los_dec = np.zeros(self.N_data)
        while i_t<self.N_data:
            dmjd = self.mjds[i_t]-timepoint
            veff_ra,veff_dec = self.veff_at(timepoint)
            if dmjd>int_step:
                integral_ra += veff_ra*int_step*day
                integral_dec += veff_dec*int_step*day
                timepoint += int_step
            else:
                integral_ra += veff_ra*dmjd*day
                integral_dec += veff_dec*dmjd*day
                timepoint = self.mjds[i_t]
                self.th_los_ra[i_t] = integral_ra
                self.th_los_dec[i_t] = integral_dec
                i_t += 1
        self.th_los_ra = self.th_los_ra / self.Deff
        self.th_los_dec = self.th_los_dec / self.Deff
        
        return self.th_los_ra,self.th_los_dec
       
class StatPoints:
    def  __init__(self):
        self.screen = []
        self.noise = False
        self.pulse_variation = False
        
    def add_screen(self,**kwargs):
        Deff = kwargs.get("Deff",1.)
        veff = kwargs.get("veff",1.)
        geometry = kwargs.get("geometry","1DFrozen")
        if geometry == "1DFrozen":
            self.screen.append(SP_1DFrozen(Deff,veff))
        elif geometry == "1DFrozenTVar":
            TVar = kwargs.get("TVar",np.array([0.]))
            self.screen.append(SP_1DFrozenTVar(Deff,veff,TVar))
        elif geometry == "1DTVar":
            N_t = kwargs.get("N_t",100)
            self.screen.append(SP_1DTVar(Deff,veff,N_t))
        elif geometry == "1DTDrift":
            slope = kwargs.get("slope",1.)
            th_Drift = kwargs.get("th_Drift",np.array([0.]))
            Drift = kwargs.get("Drift",np.array([0.]))
            self.screen.append(SP_1DTDrift(Deff,veff,th_Drift,Drift,slope))
        else:
            raise KeyError
            
    def add_noise(self,**kwargs):
        self.noise = True
        #rms of noise in units of mean signal
        self.noise_rms = kwargs.get("noise_rms",0.1)
        
    def add_pulse_variation(self,**kwargs):
        self.pulse_variation = True
        #radius of fractional variation around the mean amplitude
        self.noise_pulse = kwargs.get("noise_pulse",0.2)
            
    def compute_DS(self,t,nu):
        N_t = len(t)
        N_nu = len(nu)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_im = np.zeros(N_t*N_nu,dtype=float)
        
        for scr in self.screen:
            scr.compute(t,nu,E_real,E_im)
        E = E_real.reshape((N_t,N_nu))+1.j*E_im.reshape((N_t,N_nu))
        mean_signal = np.mean(np.abs(E))
            
        rng = np.random.default_rng(12345)
        if self.pulse_variation:
            variation = 1.-self.noise_pulse + (rng.random(N_t)-0.5)*2*self.noise_pulse
            E = E*variation[:,na]
        if self.noise:
            E_noise = (rng.standard_normal((N_t,N_nu))+rng.standard_normal((N_t,N_nu))) * self.noise_rms*mean_signal
            E = E + E_noise
            
        DS = np.abs(E)**2
        return DS
        
class SP_1DFrozen:
    """
    helper class for StatPoints
    organizing images on a simple frozen 1D screen
    """
    def __init__(self,Deff,veff):
        self.Deff = Deff
        self.veff = veff
        self.th = np.empty(0,dtype=float)
        self.mu = np.empty(0,dtype=float)
        self.ph = np.empty(0,dtype=float)
        
    def add_SP(self,th=0.,mu=0.,ph=0.):
        self.th = np.append(self.th,th)
        self.mu = np.append(self.mu,mu)
        self.ph = np.append(self.ph,ph)
        
    def compute(self,t,nu,E_real,E_im):
        N_t = len(t)
        N_nu = len(nu)
        rad_to_psi = np.sqrt(np.pi*self.Deff/v_c)
        w = np.abs(self.veff)/self.Deff * rad_to_psi
        wt = w*t
        psi = self.th * rad_to_psi
        N_th = len(psi)
        lib.SumStatPoints(N_nu, N_th, N_t, nu, self.mu, self.ph, psi, wt, E_real, E_im)
        
class SP_1DFrozenTVar:
    """
    helper class for StatPoints
    organizing images on a frozen 1D screen with temporal modulation
    """
    def __init__(self,Deff,veff,TVar):
        self.Deff = Deff
        self.veff = veff
        self.TVar = TVar
        self.th = np.empty(0,dtype=float)
        self.mu = np.empty(0,dtype=float)
        self.ph = np.empty(0,dtype=float)
        
    def add_SP(self,th=0.,mu=0.,ph=0.):
        self.th = np.append(self.th,th)
        self.mu = np.append(self.mu,mu)
        self.ph = np.append(self.ph,ph)
        
    def compute(self,t,nu,E_real,E_im):
        N_t = len(t)
        N_nu = len(nu)
        rad_to_psi = np.sqrt(np.pi*self.Deff/v_c)
        w = np.abs(self.veff)/self.Deff * rad_to_psi
        wt = w*t
        psi = self.th * rad_to_psi
        N_th = len(psi)
        lib.SumStatPoints_GlobTVar(N_nu, N_th, N_t, nu, self.mu, self.ph, psi, wt, self.TVar, E_real, E_im)
        
class SP_1DTVar:
    """
    helper class for StatPoints
    organizing images on a frequency-frozen 1D screen
    """
    def __init__(self,Deff,veff,N_t):
        self.Deff = Deff
        self.veff = veff
        self.th = np.empty(0,dtype=float)
        self.mu = np.empty(0,dtype=float)
        self.ph = np.empty(0,dtype=float)
        self.TVar = np.empty((0,N_t),dtype=float)
        
    def add_SP(self,th=0.,mu=0.,ph=0.,TVar=None):
        self.th = np.append(self.th,th)
        self.mu = np.append(self.mu,mu)
        self.ph = np.append(self.ph,ph)
        self.TVar = np.concatenate((self.TVar,TVar[na,:]),axis=0)
        
    def compute(self,t,nu,E_real,E_im):
        N_t = len(t)
        N_nu = len(nu)
        rad_to_psi = np.sqrt(np.pi*self.Deff/v_c)
        w = np.abs(self.veff)/self.Deff * rad_to_psi
        wt = w*t
        psi = self.th * rad_to_psi
        N_th = len(psi)
        lib.SumStatPoints_SingleTVar(N_nu, N_th, N_t, nu, self.mu, self.ph, psi, wt, self.TVar.flatten(), E_real, E_im)
        
class SP_1DTDrift:
    """
    helper class for StatPoints
    organizing images on a frequency-frozen 1D screen with drifting modulation of subimages
    """
    def __init__(self,Deff,veff,th_Drift,Drift,slope):
        self.Deff = Deff
        self.veff = veff
        self.th = np.empty(0,dtype=float)
        self.mu = np.empty(0,dtype=float)
        self.ph = np.empty(0,dtype=float)
        self.th_Drift = th_Drift
        self.Drift = Drift
        self.slope = slope
        
    def add_SP(self,th=0.,mu=0.,ph=0.):
        self.th = np.append(self.th,th)
        self.mu = np.append(self.mu,mu)
        self.ph = np.append(self.ph,ph)
        
    def compute(self,t,nu,E_real,E_im):
        N_t = len(t)
        N_nu = len(nu)
        rad_to_psi = np.sqrt(np.pi*self.Deff/v_c)
        w = np.abs(self.veff)/self.Deff * rad_to_psi
        wt = w*t
        psi = self.th * rad_to_psi
        psi_Drift = self.th_Drift * rad_to_psi
        N_th = len(psi)
        N_D = len(psi_Drift)
        lib.SumStatPoints_TDrift(N_nu, N_th, N_t, N_D, self.slope*rad_to_psi/w, nu, self.mu, self.ph, psi, wt, psi_Drift, self.Drift, E_real, E_im)
        
class OnTel_1DScreen:
    def  __init__(self,**kwargs):
        self.D_x = kwargs.get("D_x",1.)*pc
        self.D_s = kwargs.get("D_s",3.)*pc
        self.V_x = kwargs.get("V_x",0.)*1000.
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        self.V_s_ra = PMRA*self.D_s
        self.V_s_dec = PMDEC*self.D_s
        self.a_x = kwargs.get("a_x",0.)*np.pi/180.
        #anisotropy axes
        self.uv_x_par = np.array([np.cos(self.a_x), np.sin(self.a_x)])
        self.uv_x_ort = np.array([-np.sin(self.a_x), np.cos(self.a_x)])
        #pulsar velocity projected on second screen
        v_psr_vec = np.array([PMRA, PMDEC])*self.D_s
        self.V_s_par = np.dot(v_psr_vec, self.uv_x_par)
        self.V_s_ort = np.dot(v_psr_vec, self.uv_x_ort)
        
        if self.D_x<=0. or self.D_s<=self.D_x:
            raise ValueError
        
        self.x = np.empty(0,dtype=float)
        self.mu_x = np.empty(0,dtype=float)
        self.ph_x = np.empty(0,dtype=float)
        
        self.s = 0.
        
        self.mu_CPx = 0.
    
        self.noise = False
        self.pulse_variation = False
        
        self.D_xs = self.D_s-self.D_x
        
    def add_SP(self,th=0.,mu=1.,ph=0.):
        self.x = np.append(self.x,th*self.D_x)
        self.mu_x = np.append(self.mu_x,mu)
        self.ph_x = np.append(self.ph_x,ph)
        
    def add_CP(self,mu=1.):
        self.mu_CPx = mu
        
    def add_source_offset(self,offset):
        self.s = offset   
            
    def add_noise(self,**kwargs):
        self.noise = True
        #rms of noise in units of mean signal
        self.noise_rms = kwargs.get("noise_rms",0.1)
        
    def add_pulse_variation(self,**kwargs):
        self.pulse_variation = True
        #radius of fractional variation around the mean amplitude
        self.noise_pulse = kwargs.get("noise_pulse",0.2)
        
    def get_observer(self,mjds,psrname,telcoords):
        #get telescope velocity projected to pulsar
        if type(psrname) == str:
            psr = SkyCoord.from_name(psrname)
        else:
            psr = psrname
        rarad = psr.ra.value * np.pi/180
        decrad = psr.dec.value * np.pi/180
        
        if (type(mjds) is list) or (type(mjds) is np.ndarray):
            vtel_vec = np.zeros((2,len(mjds)))
            p_vec = np.zeros((2,len(mjds)))
            for i,mjd in enumerate(mjds):
                time = Time(mjd, format='mjd')
                pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
                vx = vel_xyz.x.to(u.m/u.s).value
                vy = vel_xyz.y.to(u.m/u.s).value
                vz = vel_xyz.z.to(u.m/u.s).value
                vtel_vec[0,i] = - vx * np.sin(rarad) + vy * np.cos(rarad)
                vtel_vec[1,i] = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
                pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
                px = pos_xyz.x.to(u.m).value
                py = pos_xyz.y.to(u.m).value
                pz = pos_xyz.z.to(u.m).value
                p_vec[0,i] = - px * np.sin(rarad) + py * np.cos(rarad)
                p_vec[1,i] = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
                
        else:
            mjd = mjds
            time = Time(mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            vtel_ra = - vx * np.sin(rarad) + vy * np.cos(rarad)
            vtel_dec = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
            vtel_vec = np.array([vtel_ra,vtel_dec])
            pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
            px = pos_xyz.x.to(u.m).value
            py = pos_xyz.y.to(u.m).value
            pz = pos_xyz.z.to(u.m).value
            p_ra =  - px * np.sin(rarad) + py * np.cos(rarad)
            p_dec = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
            p_vec = np.array([p_ra,p_dec])
            
        return vtel_vec,p_vec
        
    def compute_DS(self,t,nu,mjds,psrname,telcoords):
        N_t = len(t)
        N_nu = len(nu)
        N_x = len(self.x)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_im = np.zeros(N_t*N_nu,dtype=float)
        
        vtel_vec,p_vec = self.get_observer(mjds,psrname,telcoords)
        
        lib.SumStatPoints_1scr_PS_CP(E_real,E_im,N_t,N_nu,t,nu,N_x,self.x,self.D_x,self.D_s,self.V_x,vtel_vec[0,:],vtel_vec[1,:],p_vec[0,:],p_vec[1,:],self.V_s_ra,self.V_s_dec,self.s,self.a_x,self.mu_x,self.ph_x,self.mu_CPx)
        E = E_real.reshape((N_t,N_nu))+1.j*E_im.reshape((N_t,N_nu))
        mean_signal = np.mean(np.abs(E))
            
        rng = np.random.default_rng(12345)
        if self.pulse_variation:
            variation = 1.-self.noise_pulse + (rng.random(N_t)-0.5)*2*self.noise_pulse
            E = E*variation[:,na]
        if self.noise:
            E_noise = (rng.standard_normal((N_t,N_nu))+rng.standard_normal((N_t,N_nu))) * self.noise_rms*mean_signal
            E = E + E_noise
            
        #DS = np.abs(E)**2
        return E,p_vec
        
    def compute_SS_theo(self,t,nu,mjds,psrname,telcoords):
        vtel_vec,p_vec = self.get_observer(mjds,psrname,telcoords)
        
        
    def get_eta(self,mjd,psrname,telcoords,nu0):
        
        vtel_vec,p_vec = self.get_observer(mjd,psrname,telcoords)
        V_p_par = np.dot(vtel_vec,self.uv_x_par)
        
        Deff = self.D_x*self.D_s/self.D_xs
        
        veff = V_p_par + self.D_x/self.D_xs*self.V_s_par - self.D_s/self.D_xs*self.V_x
        
        eta = v_c*Deff/(2.*nu0**2*veff**2)
        
        return Deff,veff,eta
    
    def get_pulse(self,t_min,t_max,N_ph):
        t = np.linspace(t_min,t_max,num=N_ph,endpoint=False,dtype=float)
        dt = t[1] - t[0]
        profile = np.zeros_like(t)
        Deff = self.D_x*self.D_s/self.D_xs
        f = Deff/(2.*v_c)/self.D_x**2
        for i_x,v_x in enumerate(self.x):
            tau = f*v_x**2
            i_t = int( (tau-t[0])/dt )
            if 0<=i_t<N_ph:
                profile[i_t] += self.mu_x[i_x]
        return t,profile
    
class OnTel_2DScreen:
    def  __init__(self,**kwargs):
        self.D_x = kwargs.get("D_x",1.)*pc
        self.D_s = kwargs.get("D_s",3.)*pc
        self.V_x_ra = kwargs.get("V_x_ra",0.)*1000.
        self.V_x_dec = kwargs.get("V_x_dec",0.)*1000.
        self.psrname = kwargs.get("psrname","MISSING:psrname")
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        self.V_s_ra = PMRA*self.D_s
        self.V_s_dec = PMDEC*self.D_s
        #self.a_x = kwargs.get("a_x",0.)*np.pi/180.
        #anisotropy axes
        #self.uv_x_par = np.array([np.cos(self.a_x), np.sin(self.a_x)])
        #self.uv_x_ort = np.array([-np.sin(self.a_x), np.cos(self.a_x)])
        #pulsar velocity projected on second screen
        #v_psr_vec = np.array([PMRA, PMDEC])*self.D_s
        #self.V_s_par = np.dot(v_psr_vec, self.uv_x_par)
        #self.V_s_ort = np.dot(v_psr_vec, self.uv_x_ort)
        
        if self.D_x<=0. or self.D_s<=self.D_x:
            raise ValueError
        
        self.x_ra = np.empty(0,dtype=float)
        self.x_dec = np.empty(0,dtype=float)
        self.mu_x = np.empty(0,dtype=float)
        self.ph_x = np.empty(0,dtype=float)
        
        self.s_ra = 0.
        self.s_dec = 0.
        
        self.mu_CPx = 0.
    
        self.noise = False
        self.pulse_variation = False
        
        self.D_xs = self.D_s-self.D_x
        
    def add_SP(self,th_ra=0.,th_dec=0.,mu=1.,ph=0.):
        self.x_ra = np.append(self.x_ra,th_ra*self.D_x)
        self.x_dec = np.append(self.x_dec,th_dec*self.D_x)
        self.mu_x = np.append(self.mu_x,mu)
        self.ph_x = np.append(self.ph_x,ph)
        
    def add_CP(self,mu=1.):
        self.mu_CPx = mu
        
    def add_source_offset(self,offset_ra=0.,offset_dec=0.):
        self.s_ra = offset_ra
        self.s_dec = offset_dec
            
    def add_noise(self,**kwargs):
        self.noise = True
        #rms of noise in units of mean signal
        self.noise_rms = kwargs.get("noise_rms",0.1)
        
    def add_pulse_variation(self,**kwargs):
        self.pulse_variation = True
        #radius of fractional variation around the mean amplitude
        self.noise_pulse = kwargs.get("noise_pulse",0.2)
        
    def get_observer(self,mjds,telcoords):
        #get telescope velocity projected to pulsar
        if type(self.psrname) == str:
            psr = SkyCoord.from_name(self.psrname)
        else:
            psr = self.psrname
        rarad = psr.ra.value * np.pi/180
        decrad = psr.dec.value * np.pi/180
        
        if (type(mjds) is list) or (type(mjds) is np.ndarray):
            vtel_vec = np.zeros((2,len(mjds)))
            p_vec = np.zeros((2,len(mjds)))
            for i,mjd in enumerate(mjds):
                time = Time(mjd, format='mjd')
                pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
                vx = vel_xyz.x.to(u.m/u.s).value
                vy = vel_xyz.y.to(u.m/u.s).value
                vz = vel_xyz.z.to(u.m/u.s).value
                vtel_vec[0,i] = - vx * np.sin(rarad) + vy * np.cos(rarad)
                vtel_vec[1,i] = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
                pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
                px = pos_xyz.x.to(u.m).value
                py = pos_xyz.y.to(u.m).value
                pz = pos_xyz.z.to(u.m).value
                p_vec[0,i] = - px * np.sin(rarad) + py * np.cos(rarad)
                p_vec[1,i] = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
                
        else:
            mjd = mjds
            time = Time(mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            vtel_ra = - vx * np.sin(rarad) + vy * np.cos(rarad)
            vtel_dec = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
            vtel_vec = np.array([vtel_ra,vtel_dec])
            pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
            px = pos_xyz.x.to(u.m).value
            py = pos_xyz.y.to(u.m).value
            pz = pos_xyz.z.to(u.m).value
            p_ra =  - px * np.sin(rarad) + py * np.cos(rarad)
            p_dec = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
            p_vec = np.array([p_ra,p_dec])
            
        return vtel_vec,p_vec
        
    def compute_DS(self,t,nu,mjds,telcoords):
        N_t = len(t)
        N_nu = len(nu)
        N_x = len(self.x_ra)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_im = np.zeros(N_t*N_nu,dtype=float)
        
        vtel_vec,p_vec = self.get_observer(mjds,telcoords)
        
        lib.SumStatPoints_1scr_PS_CP_2D(E_real,E_im,N_t,N_nu,t,nu,N_x,self.x_ra,self.x_dec,self.D_x,self.D_s,self.V_x_ra,self.V_x_ra,vtel_vec[0,:],vtel_vec[1,:],p_vec[0,:],p_vec[1,:],self.V_s_ra,self.V_s_dec,self.s_ra,self.s_dec,self.mu_x,self.ph_x,self.mu_CPx)
        E = E_real.reshape((N_t,N_nu))+1.j*E_im.reshape((N_t,N_nu))
        mean_signal = np.mean(np.abs(E))
            
        rng = np.random.default_rng(12345)
        if self.pulse_variation:
            variation = 1.-self.noise_pulse + (rng.random(N_t)-0.5)*2*self.noise_pulse
            E = E*variation[:,na]
        if self.noise:
            E_noise = (rng.standard_normal((N_t,N_nu))+rng.standard_normal((N_t,N_nu))) * self.noise_rms*mean_signal
            E = E + E_noise
            
        #DS = np.abs(E)**2
        return E,p_vec
    
    def get_pulse(self,t_min,t_max,N_ph):
        t = np.linspace(t_min,t_max,num=N_ph,endpoint=False,dtype=float)
        dt = t[1] - t[0]
        profile = np.zeros_like(t)
        Deff = self.D_x*self.D_s/self.D_xs
        f = Deff/(2.*v_c)/self.D_x**2
        for i_x,v_x in enumerate(self.x):
            tau = f*v_x**2
            i_t = int( (tau-t[0])/dt )
            if 0<=i_t<N_ph:
                profile[i_t] += self.mu_x[i_x]
        return t,profile
        
class Two1DScreens:
    def  __init__(self,**kwargs):
        self.D_x = kwargs.get("D_x",1.)*pc
        self.D_y = kwargs.get("D_y",2.)*pc
        self.D_s = kwargs.get("D_s",3.)*pc
        self.V_x = kwargs.get("V_x",0.)*1000.
        self.V_y = kwargs.get("V_y",0.)*1000.
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        self.V_s_ra = PMRA*self.D_s
        self.V_s_dec = PMDEC*self.D_s
        self.a_x = kwargs.get("a_x",0.)*np.pi/180.
        self.a_y = kwargs.get("a_y",0.)*np.pi/180.
        #anisotropy axes
        self.uv_x_par = np.array([np.cos(self.a_x), np.sin(self.a_x)])
        self.uv_x_ort = np.array([-np.sin(self.a_x), np.cos(self.a_x)])
        self.uv_y_par = np.array([np.cos(self.a_y), np.sin(self.a_y)])
        self.uv_y_ort = np.array([-np.sin(self.a_y), np.cos(self.a_y)])
        #pulsar velocity projected on second screen
        v_psr_vec = np.array([PMRA, PMDEC])*self.D_s
        self.V_s_par = np.dot(v_psr_vec, self.uv_y_par)
        self.V_s_ort = np.dot(v_psr_vec, self.uv_y_ort)
        
        if self.D_x<=0. or self.D_y<=self.D_x or self.D_s<=self.D_y:
            raise ValueError
        
        self.x = np.empty(0,dtype=float)
        self.mu_x = np.empty(0,dtype=float)
        self.ph_x = np.empty(0,dtype=float)
        self.y = np.empty(0,dtype=float)
        self.mu_y = np.empty(0,dtype=float)
        self.ph_y = np.empty(0,dtype=float)
        
        self.mu_CPx = 0.
        self.mu_CPy = 0.
        
        self.p_ra = 0.
        self.p_dec = 0.
        self.s_ra = 0.
        self.s_dec = 0.
    
        self.noise = False
        self.pulse_variation = False
        
        self.D_xs = self.D_s-self.D_x
        self.D_xy = self.D_y-self.D_x
        self.D_ys = self.D_s-self.D_y
        self.c = np.cos(self.a_x-self.a_y)
        self.s = np.sin(self.a_x-self.a_y)
        
    def add_SPx(self,th=0.,mu=0.,ph=0.):
        self.x = np.append(self.x,th*self.D_x)
        self.mu_x = np.append(self.mu_x,mu)
        self.ph_x = np.append(self.ph_x,ph)
        
    def add_SPy(self,th=0.,mu=0.,ph=0.):
        self.y = np.append(self.y,th*self.D_y)
        self.mu_y = np.append(self.mu_y,mu)
        self.ph_y = np.append(self.ph_y,ph)
        
    def add_CPx(self,mu=0.):
        self.mu_CPx = mu
        
    def add_CPy(self,mu=0.):
        self.mu_CPy = mu
        
    def add_observer_offset(self,ra,dec):
        self.p_ra = ra
        self.p_dec = dec
        
    def add_source_offset(self,ra,dec):
        self.s_ra = ra
        self.s_dec = dec
            
    def add_noise(self,**kwargs):
        self.noise = True
        #rms of noise in units of mean signal
        self.noise_rms = kwargs.get("noise_rms",0.1)
        
    def add_pulse_variation(self,**kwargs):
        self.pulse_variation = True
        #radius of fractional variation around the mean amplitude
        self.noise_pulse = kwargs.get("noise_pulse",0.2)
        
    def get_vm_old(self):
        Deff_12 = self.D_s*self.D_y*self.D_x/(self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)
        #Deff_x = self.D_x*self.D_s/self.D_xs
        vm = -self.V_x + self.D_xs/self.D_ys*self.V_y*np.cos(self.a_x-self.a_y) - self.D_xy/self.D_ys*(self.V_s_ra*np.cos(self.a_x)+self.V_s_dec*np.sin(self.a_x))
        #vm = self.V_x - self.D_xs/self.D_ys*self.V_y*np.cos(self.a_x-self.a_y) - self.D_xy/self.D_ys*(self.V_s_ra*np.cos(self.a_x)+self.V_s_dec*np.sin(self.a_x))
        vm *= np.sqrt(Deff_12/(2.*v_c))/self.D_x
        
        return vm
        
    def get_vm(self,mjds,psrname):
        Deff_12 = self.D_s*self.D_y*self.D_x/(self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)
        denom = self.D_y*self.D_xs*self.c - 2.*self.D_x*self.D_ys*self.s**2*self.c
        vtel_vec = self.get_vtel(mjds,psrname)
        #Earth velocity projected on first screen
        if (type(mjds) is list) or (type(mjds) is np.ndarray):
            N_data = len(mjds)
            V_p_par = np.empty(N_data,dtype=float)
            V_p_ort = np.empty(N_data,dtype=float)
            for i in range(N_data):
                V_p_par[i] = np.dot(vtel_vec[:,i], self.uv_x_par)
                V_p_ort[i] = np.dot(vtel_vec[:,i], self.uv_x_ort)
        else:
            V_p_par = np.dot(vtel_vec, self.uv_x_par)
            V_p_ort = np.dot(vtel_vec, self.uv_x_ort)
        
        vm = -self.V_x + ( self.D_y*self.D_xs**2/self.D_ys*self.V_y + (self.D_x*self.D_xy*self.s**2-self.D_xy*self.D_y*self.D_xs/self.D_ys)*self.V_s_par - self.D_x*self.D_xy*self.s**2*self.V_s_ort - self.D_xs*self.D_xy*self.s*V_p_ort )/denom
        vm *= np.sqrt(Deff_12/(2.*v_c))/self.D_x
        
        return vm
        
    def get_vtel(self,mjds,psrname):
        #get telescope velocity projected to pulsar
        if type(psrname) == str:
            psr = SkyCoord.from_name(psrname)
        else:
            psr = psrname
        rarad = psr.ra.value * np.pi/180
        decrad = psr.dec.value * np.pi/180
        
        if (type(mjds) is list) or (type(mjds) is np.ndarray):
            vtel_vec = np.zeros((2,len(mjds)))
            for i,mjd in enumerate(mjds):
                time = Time(mjd, format='mjd')
                pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
                vx = vel_xyz.x.to(u.m/u.s).value
                vy = vel_xyz.y.to(u.m/u.s).value
                vz = vel_xyz.z.to(u.m/u.s).value
                vtel_vec[0,i] = - vx * np.sin(rarad) + vy * np.cos(rarad)
                vtel_vec[1,i] = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
        else:
            mjd = mjds
            time = Time(mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            vtel_ra = - vx * np.sin(rarad) + vy * np.cos(rarad)
            vtel_dec = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
            vtel_vec = np.array([vtel_ra,vtel_dec])
            
        return vtel_vec
    
    def get_observer(self,mjds,psrname,telcoords):
        #get telescope velocity projected to pulsar
        if type(psrname) == str:
            psr = SkyCoord.from_name(psrname)
        else:
            psr = psrname
        rarad = psr.ra.value * np.pi/180
        decrad = psr.dec.value * np.pi/180
        
        if (type(mjds) is list) or (type(mjds) is np.ndarray):
            vtel_vec = np.zeros((2,len(mjds)))
            p_vec = np.zeros((2,len(mjds)))
            for i,mjd in enumerate(mjds):
                time = Time(mjd, format='mjd')
                pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
                vx = vel_xyz.x.to(u.m/u.s).value
                vy = vel_xyz.y.to(u.m/u.s).value
                vz = vel_xyz.z.to(u.m/u.s).value
                vtel_vec[0,i] = - vx * np.sin(rarad) + vy * np.cos(rarad)
                vtel_vec[1,i] = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
                pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
                px = pos_xyz.x.to(u.m).value
                py = pos_xyz.y.to(u.m).value
                pz = pos_xyz.z.to(u.m).value
                p_vec[0,i] = - px * np.sin(rarad) + py * np.cos(rarad)
                p_vec[1,i] = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
                
        else:
            mjd = mjds
            time = Time(mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            vtel_ra = - vx * np.sin(rarad) + vy * np.cos(rarad)
            vtel_dec = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
            vtel_vec = np.array([vtel_ra,vtel_dec])
            pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
            px = pos_xyz.x.to(u.m).value
            py = pos_xyz.y.to(u.m).value
            pz = pos_xyz.z.to(u.m).value
            p_ra =  - px * np.sin(rarad) + py * np.cos(rarad)
            p_dec = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
            p_vec = np.array([p_ra,p_dec])
            
        return vtel_vec,p_vec

        
    def get_etas(self,mjd,psrname):
        
        vtel_vec = self.get_vtel(mjd,psrname)
        
        uv_x_par = np.array([np.cos(self.a_x), np.sin(self.a_x)])
        uv_x_ort = np.array([-np.sin(self.a_x), np.cos(self.a_x)])
        uv_y_par = np.array([np.cos(self.a_y), np.sin(self.a_y)])
        uv_y_ort = np.array([-np.sin(self.a_y), np.cos(self.a_y)])
        
        V_s_vec = np.array([self.V_s_ra,self.V_s_dec])
        
        Deff_x = self.D_x*self.D_s/self.D_xs
        Deff_y = self.D_y*self.D_s/self.D_ys
        
        veff_x_vec = vtel_vec + self.D_x/self.D_xs*V_s_vec
        veff_x = np.dot(veff_x_vec,uv_x_par) - self.D_s/self.D_xs*self.V_x
        veff_y_vec = vtel_vec + self.D_y/self.D_ys*V_s_vec
        veff_y = np.dot(veff_y_vec,uv_y_par) - self.D_s/self.D_ys*self.V_y
        
        nu0 = 1.4e+9
        eta_x = v_c*Deff_x/(2.*nu0**2*veff_x**2)
        eta_y = v_c*Deff_y/(2.*nu0**2*veff_y**2)
        
        V_p_par = np.dot(vtel_vec, uv_x_par)
        V_p_ort = np.dot(vtel_vec, uv_x_ort)
        #V_s_par = np.dot(V_s_vec, uv_y_par)
        V_s_ort = np.dot(V_s_vec, uv_y_ort)
        eta_xy = self.D_s*self.D_y/self.D_x*(self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)
        eta_xy /= (self.D_s*self.D_y/self.D_x*self.V_x - self.c*self.D_s*self.V_y - self.s*self.c*self.D_ys*V_p_ort - self.s*self.D_y*V_s_ort - (self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)/self.D_x*V_p_par)**2
        eta_xy *= v_c/(2.*nu0**2)
        
        return eta_x,eta_y,eta_xy
            
    def compute_DS(self,t,nu,mjds,psrname):
        N_t = len(t)
        N_nu = len(nu)
        N_x = len(self.x)
        N_y = len(self.y)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_im = np.zeros(N_t*N_nu,dtype=float)
        
        vtel_vec = self.get_vtel(mjds,psrname)
        
        #add support for specific telescopes like in single screen case
        
        #lib.SumStatPoints_2scr(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,self.x,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_x,self.mu_y,self.ph_x,self.ph_y)
        lib.SumStatPoints_2scr_CP(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,self.x,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_x,self.mu_y,self.ph_x,self.ph_y,self.mu_CPx,self.mu_CPy,self.p_ra,self.p_dec,self.s_ra,self.s_dec)
        #if self.use_controid_x:
        #    lib.SumStatPoints_CPx(E_real,E_im,N_t,N_nu,t,nu,N_y,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_CPx,self.mu_y,self.ph_y)
        E = E_real.reshape((N_t,N_nu))+1.j*E_im.reshape((N_t,N_nu))
        mean_signal = np.mean(np.abs(E))
            
        rng = np.random.default_rng(12345)
        if self.pulse_variation:
            variation = 1.-self.noise_pulse + (rng.random(N_t)-0.5)*2*self.noise_pulse
            E = E*variation[:,na]
        if self.noise:
            E_noise = (rng.standard_normal((N_t,N_nu))) * self.noise_rms*mean_signal
            E = E + E_noise
            
        DS = np.abs(E)**2
        return E,DS
    
    def compute_E_at_tel(self,t,nu,mjds,psrname,telcoords,rng_noise = np.random.default_rng(12345),rng_var = np.random.default_rng(12345)):
        N_t = len(t)
        N_nu = len(nu)
        N_x = len(self.x)
        N_y = len(self.y)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_im = np.zeros(N_t*N_nu,dtype=float)
        
        vtel_vec,p_vec = self.get_observer(mjds,psrname,telcoords)
        
        #add support for specific telescopes like in single screen case
        
        #lib.SumStatPoints_2scr(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,self.x,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_x,self.mu_y,self.ph_x,self.ph_y)
        lib.SumStatPoints_2scr_atTel_CP(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,self.x,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_x,self.mu_y,self.ph_x,self.ph_y,self.mu_CPx,self.mu_CPy,p_vec[0,:],p_vec[1,:],self.s_ra,self.s_dec)
        #if self.use_controid_x:
        #    lib.SumStatPoints_CPx(E_real,E_im,N_t,N_nu,t,nu,N_y,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_CPx,self.mu_y,self.ph_y)
        E = E_real.reshape((N_t,N_nu))+1.j*E_im.reshape((N_t,N_nu))
        mean_signal = np.mean(np.abs(E))
            
        if self.pulse_variation:
            variation = 1.-self.noise_pulse + (rng_var.random(N_t)-0.5)*2*self.noise_pulse
            E = E*variation[:,na]
        if self.noise:
            E_noise = (rng_noise.standard_normal((N_t,N_nu))) * self.noise_rms*mean_signal
            E = E + E_noise
            
        return p_vec,E
        
    def compute_SS(self,fD,tau,mjd0,psrname,nu0):
        vtel_vec = self.get_vtel(mjd0,psrname)
        V_p_par = np.dot(vtel_vec, self.uv_x_par)
        V_p_ort = np.dot(vtel_vec, self.uv_x_ort)
        V_s_vec = np.array([self.V_s_ra,self.V_s_dec])
    
        #independent screens effective quantities
        Deff1_x = self.D_x*self.D_s/self.D_xs
        Deff1_y = self.D_y*self.D_s/self.D_ys
        Veff_x_vec = vtel_vec + self.D_x/self.D_xs*V_s_vec
        Veff1_x = np.dot(Veff_x_vec,self.uv_x_par) - self.D_s/self.D_xs*self.V_x
        Veff_y_vec = vtel_vec + self.D_y/self.D_ys*V_s_vec
        Veff1_y = np.dot(Veff_y_vec,self.uv_y_par) - self.D_s/self.D_ys*self.V_y
        #interacting screens effective quantities
        denom = self.D_y*self.D_xs - self.D_x*self.D_ys*self.c**2
        Deff2_x = self.D_s*self.D_y*self.D_x/denom
        Veff2_x = -( self.D_s*self.D_y*self.V_x - self.c*self.D_s*self.D_x*self.V_y - self.s*self.c*self.D_x*self.D_ys*V_p_ort - self.s*self.D_x*self.D_y*self.V_s_ort - denom*V_p_par ) / denom
        Deff2_y = (self.D_s*self.D_xs*self.D_y**2/self.D_ys)/denom
        Veff2_y = -( self.D_s*self.D_xs*self.D_y/self.D_ys*self.V_y - self.c*self.D_y*self.D_s*self.V_x + self.s*self.D_y*self.D_xs*V_p_ort + self.s*self.c*self.D_x*self.D_y*self.V_s_ort - denom*self.D_y/self.D_ys*self.V_s_par ) / denom
        D_mix = self.c*self.D_x*self.D_y*self.D_s/denom
        #print(Deff1_x,Deff2_x,Deff1_y,Deff2_y)
        #print(Veff1_x,Veff2_x,Veff1_y,Veff2_y)
        
        N_fD = len(fD)
        N_tau = len(tau)
        dfD = np.diff(fD).mean()
        dtau = np.diff(tau).mean()
        SS = np.zeros((N_fD,N_tau),dtype=float)
        N_x = len(self.x)
        N_y = len(self.y)
        th_x = self.x/self.D_x
        th_y = self.y/self.D_y
        for i_x in range(N_x):
            for i_y in range(N_y):
                v_tau = Deff2_x/(2.*v_c)*th_x[i_x]**2 + Deff2_y/(2.*v_c)*th_y[i_y]**2 - D_mix/v_c*th_x[i_x]*th_y[i_y]
                v_fD = -nu0/v_c*Veff2_x*th_x[i_x] - nu0/v_c*Veff2_y*th_y[i_y]
                
                i_tau = int(np.rint((v_tau-tau[0])/dtau))
                i_fD = int(np.rint((v_fD-fD[0])/dfD))
                
                if (0<=i_tau<N_tau) and (0<=i_fD<N_fD):
                    SS[i_fD,i_tau] += self.mu_x[i_x]*self.mu_y[i_y]
        for i_x in range(N_x):
            v_tau = Deff1_x/(2.*v_c)*th_x[i_x]**2
            v_fD = -nu0/v_c*Veff1_x*th_x[i_x]
            i_tau = int(np.rint((v_tau-tau[0])/dtau))
            i_fD = int(np.rint((v_fD-fD[0])/dfD))
            
            if (0<=i_tau<N_tau) and (0<=i_fD<N_fD):
                SS[i_fD,i_tau] += self.mu_x[i_x]*self.mu_CPy
        for i_y in range(N_y):
            v_tau = Deff1_y/(2.*v_c)*th_y[i_y]**2
            v_fD = -nu0/v_c*Veff1_y*th_y[i_y]
            i_tau = int(np.rint((v_tau-tau[0])/dtau))
            i_fD = int(np.rint((v_fD-fD[0])/dfD))
            
            if (0<=i_tau<N_tau) and (0<=i_fD<N_fD):
                SS[i_fD,i_tau] += self.mu_y[i_y]*self.mu_CPx
                
        #print(self.mu_CPx,self.mu_CPy)
        #print("eta_x={0}".format(v_c*Deff2_x/(2.*nu0**2*Veff2_x**2)))
        return SS
        
        
    # def compute_E_at_2ndSCR(self,t,nu,stau):
        # Deff_12 = self.D_s*self.D_y*self.D_x/(self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)
        # Deff_x = self.D_x*self.D_s/self.D_xs
        # x_par = -self.D_x*stau*np.sqrt(2.*v_c/Deff_12)*np.cos(self.a_y-self.a_x)
        # V_x_par = self.V_x*np.cos(self.a_y-self.a_x)
        # V_s_vec = np.array([self.V_s_ra,self.V_s_dec])
        # uv_y_par = np.array([np.cos(self.a_y), np.sin(self.a_y)])
        # V_s_par = np.dot(V_s_vec,uv_y_par)
        
        # N_t = len(t)
        # N_nu = len(nu)
        # N_y = len(self.y)
        # N_stau = len(stau)
        # N_x = N_stau
        # E_real = np.zeros(N_t*N_nu*N_x,dtype=float)
        # E_im = np.zeros(N_t*N_nu*N_x,dtype=float)
        
        # lib.SumStatPoints_at_2ndSCR(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,x_par,self.y,self.D_x,self.D_y,self.D_s,V_x_par,self.V_y,V_s_par,self.mu_y,self.ph_y)
        # E = E_real.reshape((N_t,N_nu,N_x))+1.j*E_im.reshape((N_t,N_nu,N_x))
        
        # return E
        
    def compute_E_at_2ndSCR(self,t,nu,stau,mjds,psrname):
        Deff_12 = self.D_s*self.D_y*self.D_x/(self.D_y*self.D_xs-self.D_x*self.D_ys*self.c**2)
        Deff_x = self.D_x*self.D_s/self.D_xs
        x = -self.D_x*stau*np.sqrt(2.*v_c/Deff_12)
        
        N_t = len(t)
        N_nu = len(nu)
        N_x = len(x)
        N_y = len(self.y)
        E_real = np.zeros(N_t*N_nu*N_x,dtype=float)
        E_im = np.zeros(N_t*N_nu*N_x,dtype=float)
        
        vtel_vec = self.get_vtel(mjds,psrname)
        
        lib.SumStatPoints_at_2ndSCR(E_real,E_im,N_t,N_nu,t,nu,N_x,N_y,x,self.y,self.D_x,self.D_y,self.D_s,self.V_x,self.V_y,vtel_vec[0,:],vtel_vec[1,:],self.V_s_ra,self.V_s_dec,self.a_x,self.a_y,self.mu_y,self.ph_y)
        E = E_real.reshape((N_t,N_nu,N_x))+1.j*E_im.reshape((N_t,N_nu,N_x))
        
        return E
    
class Two1DScreens_obsUnits(Two1DScreens):
    def  __init__(self,**kwargs):
        self.D_x = kwargs.get("D_x",1.)*pc
        self.D_y = kwargs.get("D_y",2.)*pc
        self.D_s = kwargs.get("D_s",3.)*pc
        self.V_x = kwargs.get("V_x",0.)*1000.
        self.V_y = kwargs.get("V_y",0.)*1000.
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        self.V_s_ra = PMRA*self.D_s
        self.V_s_dec = PMDEC*self.D_s
        self.a_x = kwargs.get("a_x",0.)*np.pi/180.
        self.a_y = kwargs.get("a_y",0.)*np.pi/180.
        #anisotropy axes
        self.uv_x_par = np.array([np.cos(self.a_x), np.sin(self.a_x)])
        self.uv_x_ort = np.array([-np.sin(self.a_x), np.cos(self.a_x)])
        self.uv_y_par = np.array([np.cos(self.a_y), np.sin(self.a_y)])
        self.uv_y_ort = np.array([-np.sin(self.a_y), np.cos(self.a_y)])
        #pulsar velocity projected on second screen
        v_psr_vec = np.array([PMRA, PMDEC])*self.D_s
        self.V_s_par = np.dot(v_psr_vec, self.uv_y_par)
        self.V_s_ort = np.dot(v_psr_vec, self.uv_y_ort)
        
        if self.D_x<=0. or self.D_y<=self.D_x or self.D_s<=self.D_y:
            raise ValueError
        
        self.x = np.empty(0,dtype=float)
        self.mu_x = np.empty(0,dtype=float)
        self.ph_x = np.empty(0,dtype=float)
        self.y = np.empty(0,dtype=float)
        self.mu_y = np.empty(0,dtype=float)
        self.ph_y = np.empty(0,dtype=float)
        
        self.mu_CPx = 0.
        self.mu_CPy = 0.
    
        self.noise = False
        self.pulse_variation = False
        
        self.D_xs = self.D_s-self.D_x
        self.D_xy = self.D_y-self.D_x
        self.D_ys = self.D_s-self.D_y
        self.c = np.cos(self.a_x-self.a_y)
        self.s = np.sin(self.a_x-self.a_y)
        
    def add_SPx(self,th=0.,mu=0.,ph=0.):
        self.x = np.append(self.x,th*self.D_x)
        self.mu_x = np.append(self.mu_x,mu)
        self.ph_x = np.append(self.ph_x,ph)
        
    def add_SPy(self,th=0.,mu=0.,ph=0.):
        self.y = np.append(self.y,th*self.D_y)
        self.mu_y = np.append(self.mu_y,mu)
        self.ph_y = np.append(self.ph_y,ph)
        
class Evolution_Two1DScreens:
    def  __init__(self,mjds,psrname,telcoords="",include_earth_rotation_in_veff=False):
        #load pulsar
        if type(psrname) == str:
            pulsar = SkyCoord.from_name(psrname)
        else:
            pulsar = psrname
        self.rarad = pulsar.ra.value * np.pi/180
        self.decrad = pulsar.dec.value * np.pi/180
        
        vtel_ra = np.zeros_like(mjds)
        vtel_dec = np.zeros_like(mjds)
        for i,v_mjd in enumerate(mjds):
            time = Time(v_mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            vtel_ra[i] = - vx * np.sin(self.rarad) + vy * np.cos(self.rarad)
            vtel_dec[i] = - vx * np.sin(self.decrad) * np.cos(self.rarad) - vy * np.sin(self.decrad) * np.sin(self.rarad) + vz * np.cos(self.decrad)
        self.vtel_vec = np.array([vtel_ra,vtel_dec])
        self.mjds = mjds
        
        if telcoords=="":
            self.p_vec = np.zeros((2,len(self.mjds)),dtype=float)
            self.p_w = np.zeros((len(self.mjds)),dtype=float)
        else:
            p_ra = np.zeros_like(self.mjds)
            p_dec = np.zeros_like(self.mjds)
            p_w = np.zeros_like(self.mjds)
            vtel_ra = np.zeros_like(mjds)
            vtel_dec = np.zeros_like(mjds)
            for i,v_mjd in enumerate(self.mjds):
                time = Time(v_mjd, format='mjd')
                pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
                px = pos_xyz.x.to(u.m).value
                py = pos_xyz.y.to(u.m).value
                pz = pos_xyz.z.to(u.m).value
                p_ra[i] = - px * np.sin(self.rarad) + py * np.cos(self.rarad)
                p_dec[i] = - px * np.sin(self.decrad) * np.cos(self.rarad) - py * np.sin(self.decrad) * np.sin(self.rarad) + pz * np.cos(self.decrad)
                p_w[i] = px * np.cos(self.decrad) * np.cos(self.rarad) + py * np.cos(self.decrad) * np.sin(self.rarad) + pz * np.sin(self.decrad)
                vx = vel_xyz.x.to(u.m/u.s).value
                vy = vel_xyz.y.to(u.m/u.s).value
                vz = vel_xyz.z.to(u.m/u.s).value
                vtel_ra[i] = - vx * np.sin(self.rarad) + vy * np.cos(self.rarad)
                vtel_dec[i] = - vx * np.sin(self.decrad) * np.cos(self.rarad) - vy * np.sin(self.decrad) * np.sin(self.rarad) + vz * np.cos(self.decrad)
            #apply correction to velocity from Earth's rotation (careful: varies during observation and is already included in p(t) where it is used)
            if include_earth_rotation_in_veff:
                self.vtel_vec = self.vtel_vec + np.array([vtel_ra,vtel_dec])
            #save position of telescope
            self.p_vec = np.array([p_ra,p_dec])
            self.p_w = p_w
        
        self.library = {}
            
    def compute(self,**kwargs):
        #load free parameters
        a_x = kwargs.get("a_x",0.)*np.pi/180.
        a_y = kwargs.get("a_y",0.)*np.pi/180.
        D_x = kwargs.get("D_x",1.)*pc
        D_y = kwargs.get("D_y",2.)*pc
        D_s = kwargs.get("D_s",3.)*pc
        V_x = kwargs.get("V_x",0.)*1000.
        V_y = kwargs.get("V_y",0.)*1000.
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        if D_x<=0. or D_y<=D_x or D_s<=D_y:
            raise ValueError
        #pulsar velocity vector
        V_s_ra = PMRA*D_s
        V_s_dec = PMDEC*D_s
        V_s_vec = np.array([V_s_ra,V_s_dec])
        #anisotropy axes
        uv_x_par = np.array([np.cos(a_x), np.sin(a_x)])
        uv_x_ort = np.array([-np.sin(a_x), np.cos(a_x)])
        uv_y_par = np.array([np.cos(a_y), np.sin(a_y)])
        uv_y_ort = np.array([-np.sin(a_y), np.cos(a_y)])
        #observer velocity projected on first screen
        V_p_par = np.dot(np.swapaxes(self.vtel_vec,0,1), uv_x_par)
        V_p_ort = np.dot(np.swapaxes(self.vtel_vec,0,1), uv_x_ort)
        #pulsar velocity projected on second screen
        V_s_par = np.dot(V_s_vec, uv_y_par)
        V_s_ort = np.dot(V_s_vec, uv_y_ort)
        #derived quantities
        D_xs = D_s-D_x
        #D_xy = D_y-D_x
        D_ys = D_s-D_y
        c = np.cos(a_x-a_y)
        s = np.sin(a_x-a_y)
        #dictionary of results
        zetas = {}
        #independent screens effective quantities
        Deff1_x = D_x*D_s/D_xs
        Deff1_y = D_y*D_s/D_ys
        Veff_x_vec = self.vtel_vec + D_x/D_xs*V_s_vec[:,na]
        Veff1_x = np.dot(np.swapaxes(Veff_x_vec,0,1),uv_x_par) - D_s/D_xs*V_x
        Veff_y_vec = self.vtel_vec + D_y/D_ys*V_s_vec[:,na]
        Veff1_y = np.dot(np.swapaxes(Veff_y_vec,0,1),uv_y_par) - D_s/D_ys*V_y
        zetas.update({"Deff1_x":Deff1_x,"Veff1_x":Veff1_x,"Deff1_y":Deff1_y,"Veff1_y":Veff1_y,"Veff1_x_vec":np.swapaxes(Veff_x_vec,0,1)})
        #interacting screens effective quantities
        Deff2_x = D_s*D_y*D_x/( D_y*D_xs - D_x*D_ys*c**2 )
        Veff2_x = -( D_s*D_y*V_x - c*D_s*D_x*V_y - s*c*D_x*D_ys*V_p_ort - s*D_x*D_y*V_s_ort - (D_y*D_xs-D_x*D_ys*c**2)*V_p_par ) / ( D_y*D_xs - D_x*D_ys*c**2 )
        Deff2_y = (D_s*D_xs*D_y**2/D_ys)/( D_y*D_xs - D_x*D_ys*c**2 )
        Veff2_y = -( D_s*D_xs*D_y/D_ys*V_y - c*D_y*D_s*V_x + s*D_y*D_xs*V_p_ort + s*c*D_x*D_y*V_s_ort - (D_y*D_xs-D_x*D_ys*c**2)*D_y/D_ys*V_s_par ) / ( D_y*D_xs - D_x*D_ys*c**2 )
        Vmod = -V_x + ( D_s*D_xs*V_y + (D_x*D_ys*c**2-D_xs*D_y)*V_s_par + D_x*D_ys*s*c*V_s_ort + D_xs*D_ys*s*V_p_ort ) / (D_ys*D_s*c)
        #print("individual terms: Vx:{0} Vy:{1} Vsp:{2} Vso:{3} Vpo:{4}".format(-V_x, D_s*D_xs*V_y/(D_ys*D_s*c), (D_x*D_ys*c**2-D_xs*D_y)*V_s_par/(D_ys*D_s*c), D_x*D_ys*s*c*V_s_ort/(D_ys*D_s*c), D_xs*D_ys*s*V_p_ort/(D_ys*D_s*c)))
        #Vmod = -V_x + ( D_y*D_xs**2/D_ys*V_y + (D_x*D_xy*s**2-D_xy*D_y*D_xs/D_ys)*V_s_par - D_x*D_xy*s**2*V_s_ort - D_xs*D_xy*s*V_p_ort ) / ( D_y*D_xs*c - 2.*D_x*D_ys*s**2*c )
        D_mix = c*D_x*D_y*D_s/(D_y*D_xs - D_x*D_ys*c**2)
        zetas.update({"Deff2_x":Deff2_x,"Veff2_x":Veff2_x,"Deff2_y":Deff2_y,"D_mix":D_mix,"Veff2_y":Veff2_y})
        #arc parameters
        zeta1_x = np.sqrt(1./(2.*v_c*Deff1_x))*np.abs(Veff1_x)
        zeta1_y = np.sqrt(1./(2.*v_c*Deff1_y))*np.abs(Veff1_y)
        zeta2_x = np.sqrt(1./(2.*v_c*Deff2_x))*np.abs(Veff2_x)
        zeta2_y = np.sqrt(1./(2.*v_c*Deff2_y))*np.abs(Veff2_y)
        zeta2_m = np.sqrt(Deff2_x/(2.*v_c))/D_x*Vmod
        zeta2_fx = np.sqrt(Deff2_x/(2.*v_c))*np.abs((Deff2_y*Veff2_x + D_mix*Veff2_y)/(Deff2_y*Deff2_x-D_mix**2))
        zeta2_fy = np.sqrt(Deff2_y/(2.*v_c))*np.abs((Deff2_x*Veff2_y + D_mix*Veff2_x)/(Deff2_x*Deff2_y-D_mix**2))
        zetas.update({"zeta1_x":zeta1_x,"zeta1_y":zeta1_y,"zeta2_x":zeta2_x,"zeta2_y":zeta2_y,"zeta2_m":zeta2_m,"zeta2_fx":zeta2_fx,"zeta2_fy":zeta2_fy})
        #Compute time shift relative to center of Earth
        p_par = np.dot(np.swapaxes(self.p_vec,0,1), uv_x_par)
        p_ort = np.dot(np.swapaxes(self.p_vec,0,1), uv_x_ort)
        peff_x = p_par + (s*c*D_x*D_ys)/(D_y*D_xs-D_x*D_y*c**2)*p_ort
        Dt_2x = peff_x/Veff2_x - self.p_w/v_c
        peff_y = -(s*D_y*D_xs)/(D_y*D_xs-D_x*D_y*c**2)*p_ort
        Dt_2y = peff_y/Veff2_y
        zetas.update({"Dt_2x":Dt_2x,"Dt_2y":Dt_2y})
        #return as dictionary
        return zetas
    
    def _get_telescope(self,telcoords,name=""):
        p_ra = np.zeros_like(self.mjds)
        p_dec = np.zeros_like(self.mjds)
        for i,v_mjd in enumerate(self.mjds):
            time = Time(v_mjd, format='mjd')
            pos_xyz, vel_xyz = telcoords.get_gcrs_posvel(time)
            px = pos_xyz.x.to(u.m).value
            py = pos_xyz.y.to(u.m).value
            pz = pos_xyz.z.to(u.m).value
            p_ra[i] = - px * np.sin(self.rarad) + py * np.cos(self.rarad)
            p_dec[i] = - px * np.sin(self.decrad) * np.cos(self.rarad) - py * np.sin(self.decrad) * np.sin(self.rarad) + pz * np.cos(self.decrad)
        p_vec = np.array([p_ra,p_dec])
        self.library.update({name:p_vec})
        return p_vec
        
    
    def compute_Dt(self,telcoords,**kwargs):
        a_x = kwargs.get("a_x",0.)*np.pi/180.
        Veff = kwargs.get("Veff",1.)*1000.
        name = kwargs.get("name","")
        correct_for_2scr = kwargs.get("correct_for_2scr",False)
        
        if name in self.library and name!="":
            p_vec = self.library[name]
        else:
            p_vec = self._get_telescope(telcoords,name)
        
        #zetas = self.compute(kwargs)
        #Veff = zetas["Veff1_x"] #try "Veff2_x" for impact of second screen
        
        uv_x_par = np.array([np.cos(a_x), np.sin(a_x)])
        p_par = np.dot(np.swapaxes(p_vec,0,1), uv_x_par)
        
        if correct_for_2scr:
            D_x = kwargs.get("D_x",1.)*pc
            D_y = kwargs.get("D_y",2.)*pc
            D_s = kwargs.get("D_s",3.)*pc
            a_y = kwargs.get("a_y",0.)*np.pi/180.
            
            uv_x_ort = np.array([-np.sin(a_x), np.cos(a_x)])
            p_ort = np.dot(np.swapaxes(p_vec,0,1), uv_x_ort)
            
            #derived quantities
            D_xs = D_s-D_x
            #D_xy = D_y-D_x
            D_ys = D_s-D_y
            c = np.cos(a_x-a_y)
            s = np.sin(a_x-a_y)
            
            peff = p_par + (s*c*D_x*D_ys)/(D_y*D_xs-D_x*D_y*c**2)*p_ort
            Dt = peff/Veff
        else:
            Dt = p_par/Veff
        
        return Dt
    
class Evolution_One1DScreen(Evolution_Two1DScreens):
    def compute(self,**kwargs):
        #load free parameters
        a_x = kwargs.get("a_x",0.)*np.pi/180.
        D_x = kwargs.get("D_x",1.)*pc
        V_x = kwargs.get("V_x",0.)*1000.
        D_s = kwargs.get("D_s",3.)*pc
        PMRA = kwargs.get("PMRA",0.)*mas/year
        PMDEC = kwargs.get("PMDEC",0.)*mas/year
        if D_x<=0. or D_s<=D_x:
            raise ValueError
        #pulsar velocity vector
        V_s_ra = PMRA*D_s
        V_s_dec = PMDEC*D_s
        V_s_vec = np.array([V_s_ra,V_s_dec])
        #anisotropy axis
        uv_x_par = np.array([np.cos(a_x), np.sin(a_x)])
        uv_x_ort = np.array([-np.sin(a_x), np.cos(a_x)])
        #derived quantities
        D_xs = D_s-D_x
        #dictionary of results
        zetas = {}
        #independent screens effective quantities
        Deff1_x = D_x*D_s/D_xs
        Veff_x_vec = self.vtel_vec + D_x/D_xs*V_s_vec[:,na]
        Veff1_x = np.dot(np.swapaxes(Veff_x_vec,0,1),uv_x_par) - D_s/D_xs*V_x
        zetas.update({"Deff1_x":Deff1_x,"Veff1_x":Veff1_x,"Veff_x_vec":np.swapaxes(Veff_x_vec,0,1)})
        #arc parameters
        zeta1_x = np.sqrt(1./(2.*v_c*Deff1_x))*np.abs(Veff1_x)
        zetas.update({"zeta1_x":zeta1_x})
        #Compute time shift (first screen) relative to center of Earth
        p_par = np.dot(np.swapaxes(self.p_vec,0,1), uv_x_par)
        Dt_1x = p_par/Veff1_x - self.p_w/v_c
        zetas.update({"Dt_1x":Dt_1x})
        #return as dictionary
        return zetas
        
        
class DM_integration_1D:
    def __init__(self,dDM):
        self.dDM = dDM
        self.f_DM = 1./(4.*np.pi*eps0)*e**2/(me*v_c)*pc_per_cm3
        
    def set_pulsar(self,D_s,PMRA,PMDEC,skycoords):
        V_s_ra = PMRA*D_s
        V_s_dec = PMDEC*D_s
        self.V_s_vec = np.array([V_s_ra,V_s_dec])
        self.D_s = D_s
        self.skycoords = skycoords
    
    def set_screen(self,D_x,a_x,V_x,th_min,th_max):
        self.D_x = D_x
        #self.a_x = a_x
        self.V_x = V_x
        self.th_min = th_min
        self.th_max = th_max
        self.uv_x_par = np.array([np.cos(a_x), np.sin(a_x)])
    
    def set_observer(self,mjd0,t,nu,telcoords):
        self.mjds = mjd0 + t/day
        self.t = t
        self.nu = nu
        self.telcoords = telcoords
        
    def _compute_observer(self):
        print("Internally computing observer")
        #telescope projected velocity and location
        rarad = self.skycoords.ra.value * np.pi/180
        decrad = self.skycoords.dec.value * np.pi/180
        vtel_ra = np.zeros_like(self.mjds)
        vtel_dec = np.zeros_like(self.mjds)
        p_ra = np.zeros_like(self.mjds)
        p_dec = np.zeros_like(self.mjds)
        bar = progressbar.ProgressBar(maxval=len(self.mjds), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i,v_mjd in enumerate(self.mjds):
            bar.update(i)
            time = Time(v_mjd, format='mjd')
            pos_xyz, vel_xyz = get_body_barycentric_posvel('earth', time)
            vx = vel_xyz.x.to(u.m/u.s).value
            vy = vel_xyz.y.to(u.m/u.s).value
            vz = vel_xyz.z.to(u.m/u.s).value
            pos_xyz, vel_xyz = self.telcoords.get_gcrs_posvel(time)
            px = pos_xyz.x.to(u.m).value
            py = pos_xyz.y.to(u.m).value
            pz = pos_xyz.z.to(u.m).value
            #vx += vel_xyz.x.value
            #vy += vel_xyz.y.value
            #vz += vel_xyz.z.value
            vtel_ra[i] = - vx * np.sin(rarad) + vy * np.cos(rarad)
            vtel_dec[i] = - vx * np.sin(decrad) * np.cos(rarad) - vy * np.sin(decrad) * np.sin(rarad) + vz * np.cos(decrad)
            p_ra[i] =  - px * np.sin(rarad) + py * np.cos(rarad)
            p_dec[i] = - px * np.sin(decrad) * np.cos(rarad) - py * np.sin(decrad) * np.sin(rarad) + pz * np.cos(decrad)
        bar.finish()
        self.V_p_vec = np.swapaxes(np.array([vtel_ra,vtel_dec]),0,1)
        self.p_vec = np.swapaxes(np.array([p_ra,p_dec]),0,1)
        
    def _compute_screen(self,tol):
        print("Internally computing screen")
        D_xs = self.D_s-self.D_x
        self.Deff = self.D_x*self.D_s/D_xs
        self.Veff = -self.D_s/D_xs*self.V_x + np.dot(self.V_p_vec,self.uv_x_par) + self.D_x/D_xs*np.dot(self.V_s_vec,self.uv_x_par)
        self.p = np.dot(self.p_vec,self.uv_x_par)
        
        th_big = np.max([np.abs(self.th_min),np.abs(self.th_max)])
        dth_max = tol/(np.max(self.nu)/v_c*self.Deff*th_big)
        N_DM = len(self.dDM)
        th_in = np.linspace(self.th_min,self.th_max,num=N_DM,endpoint=True)
        dth_in = np.diff(th_in).mean()
        N_th = int(np.rint(dth_in/dth_max)*N_DM)
        print("Interpolating to {0} instead of {1} steps of DM.".format(N_th,N_DM))
        self.int_dDM = interp1d(th_in,self.dDM, kind='cubic')
        self.thetas = np.linspace(self.th_min,self.th_max,num=N_th,endpoint=True)
        
        
    def get_phases(self,tol):
        self._compute_observer()
        self._compute_screen(tol)
        
        phi_disp = -self.f_DM*self.int_dDM(self.thetas)/np.mean(self.nu)
        phi_delay = np.pi*self.Deff/v_c*self.thetas**2*np.mean(self.nu)
        #phi_Doppler = -2.*np.pi/v_c*(self.Veff*self.t+self.p)
        
        return self.thetas,phi_disp,phi_delay
    
    def _integrate_sample(self,inds_t,inds_nu,DMgrad_max):
        dth = np.diff(self.thetas).mean()
        N_th = len(self.thetas)
        dDM = self.int_dDM(self.thetas)
        nu_mean = np.mean(self.nu)
        mean_Phi = -self.f_DM*dDM/nu_mean + np.pi*self.Deff/v_c*self.thetas**2*nu_mean -2.*np.pi/v_c*np.mean(self.Veff*self.t+self.p)*nu_mean*self.thetas
        DMgrad = np.diff(mean_Phi)/dth
        if np.abs(DMgrad[0])<DMgrad_max:
            inds_th = [0]
        else:
            inds_th = []
        bar = progressbar.ProgressBar(maxval=N_th-1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_th in range(N_th-1):
            bar.update(i_th)
            if np.abs(DMgrad[i_th])<DMgrad_max:
                inds_th.append(i_th+1)
        bar.finish()
        print("Fraction kept: {0}".format(len(inds_th)/N_th))
        
        phi_disp = -self.f_DM*dDM[inds_th]
        phi_delay = np.pi*self.Deff/v_c*self.thetas[inds_th]**2
        phi_Doppler = -2.*np.pi/v_c*(self.Veff[inds_t]*self.t[inds_t]+self.p[inds_t])
        
        N_t = len(inds_t)
        N_nu = len(inds_nu)
        N_th = len(inds_th)
        E = np.empty((N_t,N_nu),dtype=complex)
        bar = progressbar.ProgressBar(maxval=N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        i_bar = 0
        for i_nu in inds_nu:
            bar.update(i_bar)
            v_nu = self.nu[i_nu]
            E_real = np.zeros((N_t),dtype='float64')
            E_im = np.zeros((N_t),dtype='float64')
            lib.DMI_stripe(N_t,N_th,v_nu,self.thetas[inds_th],phi_disp,phi_delay,phi_Doppler,E_real,E_im)
            E[:,i_bar] = (E_real+1.j*E_im)*dth
            i_bar += 1
        bar.finish()
        return E
    
    def _integrate(self,DMgrad_max):
        dth = np.diff(self.thetas).mean()
        N_th = len(self.thetas)
        dDM = self.int_dDM(self.thetas)
        nu_mean = np.mean(self.nu)
        mean_Phi = -self.f_DM*dDM/nu_mean + np.pi*self.Deff/v_c*self.thetas**2*nu_mean -2.*np.pi/v_c*np.mean(self.Veff*self.t+self.p)*nu_mean*self.thetas
        DMgrad = np.diff(mean_Phi)/dth
        if np.abs(DMgrad[0])<DMgrad_max:
            inds_th = [0]
        else:
            inds_th = []
        for i_th in range(N_th-1):
            if np.abs(DMgrad[i_th])<DMgrad_max:
                inds_th.append(i_th+1)
        print("Fraction kept: {0}".format(len(inds_th)/N_th))
        
        phi_disp = -self.f_DM*dDM[inds_th]
        phi_delay = np.pi*self.Deff/v_c*self.thetas[inds_th]**2
        phi_Doppler = -2.*np.pi/v_c*(self.Veff*self.t+self.p)
        
        N_t = len(self.t)
        N_nu = len(self.nu)
        N_th = len(inds_th)
        E = np.empty((N_t,N_nu),dtype=complex)
        bar = progressbar.ProgressBar(maxval=N_nu, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for i_nu in range(N_nu):
            bar.update(i_nu)
            v_nu = self.nu[i_nu]
            E_real = np.zeros((N_t),dtype='float64')
            E_im = np.zeros((N_t),dtype='float64')
            lib.DMI_stripe(N_t,N_th,v_nu,self.thetas[inds_th],phi_disp,phi_delay,phi_Doppler,E_real,E_im)
            E[:,i_nu] = (E_real+1.j*E_im)*dth
        bar.finish()
        return E
    
    def compute_numInt(self,tol,acc):
        print("Computing Observer")
        self._compute_observer()
        print("Computing Screen")
        self._compute_screen(tol)
        
        print("Computing Stationary Phase Approximation")
        
        N_t = len(self.t)
        N_nu = len(self.nu)
        
        inds_t = [0,1,2,3,4,N_t-5,N_t-4,N_t-3,N_t-2,N_t-1]
        inds_nu = [0,1,2,3,4,N_nu-5,N_nu-4,N_nu-3,N_nu-2,N_nu-1]
        
        E_ref = self._integrate_sample(inds_t,inds_nu,np.inf)
        v_ref = np.mean(np.abs(E_ref))
        
        DMgrad_max = 1.e+11
        upper = 0.
        lower = 0.
        while True:
            E_test = self._integrate_sample(inds_t,inds_nu,DMgrad_max)
            relerr = np.mean(np.abs((E_ref-E_test)/v_ref))
            print("{0} at {1}".format(relerr,DMgrad_max))
            if relerr>acc:
                lower = DMgrad_max
                if upper == 0.:
                    DMgrad_max = 10.*DMgrad_max
                else:
                    DMgrad_max = (upper+DMgrad_max)/2.
            elif relerr<0.9*acc:
                upper = DMgrad_max
                DMgrad_max = (lower+DMgrad_max)/2.
            else:
                print("Optimal max gradient found: {0}".format(DMgrad_max))
                break
                
        E = self._integrate(DMgrad_max)
        return E
        
    def compute_numInt_fraction(self,tol,fraction):
        print("Computing Observer")
        self._compute_observer()
        print("Computing Screen")
        self._compute_screen(tol)
        
        dth = np.diff(self.thetas).mean()
        N_th = len(self.thetas)
        dDM = self.int_dDM(self.thetas)
        nu_mean = np.mean(self.nu)
        mean_Phi = -self.f_DM*dDM/nu_mean + np.pi*self.Deff/v_c*self.thetas**2*nu_mean -2.*np.pi/v_c*np.mean(self.Veff*self.t+self.p)*nu_mean*self.thetas
        DMgrad = np.sort(np.abs(np.diff(mean_Phi)/dth))
        assert DMgrad[0]<DMgrad[-1]
        i_max = int(np.rint(fraction*N_th-1))
        
        DMgrad_max = DMgrad[i_max]
                
        E = self._integrate(DMgrad_max)
        return E
        
class DM2D_sinc:
    def __init__(self,DM,stau_x,stau_y):
        self.M = f_DM*DM/(2.*np.pi)
        self.stau_x = stau_x
        self.stau_y = stau_y
        self.dstau = stau_x[1]-stau_x[0]
        assert self.dstau==stau_y[1]-stau_y[0] #more testing required for asymmetric axes
        self.M_dx,self.M_dy = np.gradient(self.M,self.dstau)
        
    def compute(self,t,nu,zeta_x,zeta_y):
        N_t = len(t)
        N_nu = len(nu)
        E_real = np.zeros(N_t*N_nu,dtype=float)
        E_imag = np.zeros(N_t*N_nu,dtype=float)
        
        N_x = len(self.stau_x)
        N_y = len(self.stau_y)
        
        lib.DM2D_sinc(E_real,E_imag,N_t,N_nu,N_x,N_y,zeta_x,zeta_y,t,nu,self.stau_x,self.stau_y,self.M.flatten(),self.M_dx.flatten(),self.M_dy.flatten())
                
        E = E_real.reshape((N_t,N_nu))+1.0j*E_imag.reshape((N_t,N_nu))
        
        return E
        
