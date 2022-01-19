////////////////////////////////////////////////////////////////////////

// g++ -Wall -O2 -fopenmp -shared -Wl,-soname,lib_scinter -o lib_scinter.so -fPIC lib_scinter.cpp

#include <iostream>
#include <cmath>
#include <random>
#include <vector>

using namespace std;

extern "C"
void  NuT (int N_t, int N_nu, int N_fD, double *tt, double *nu, double* fD, double* DS, double *hSS_real, double *hSS_im)
{
	//tell c++ how to read numpy arrays
	#define  DYNSPEC(i_t,i_nu)  DS[(i_t)*N_nu + (i_nu)]
	#define  REAL(i_fD,i_nu)  hSS_real[(i_fD)*N_nu + (i_nu)]
	#define  IMAG(i_fD,i_nu)  hSS_im[(i_fD)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_fD = 0; i_fD < N_fD; i_fD++){
				phase = -2.*M_PI*fD[i_fD]*tt[i_t]*nu[i_nu];
				REAL(i_fD,i_nu) += DYNSPEC(i_t,i_nu)*cos(phase);
				IMAG(i_fD,i_nu) += DYNSPEC(i_t,i_nu)*sin(phase);
			}
		}
	}
		
	#undef  DYNSPEC
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  ENuT (int N_t, int N_nu, int N_fD, double *tt, double *nu, double* fD, double* E_real, double* E_im, double *hWF_real, double *hWF_im)
{
	//tell c++ how to read numpy arrays
	#define  E_REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  E_IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	#define  REAL(i_fD,i_nu)  hWF_real[(i_fD)*N_nu + (i_nu)]
	#define  IMAG(i_fD,i_nu)  hWF_im[(i_fD)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_fD = 0; i_fD < N_fD; i_fD++){
				phase = -2.*M_PI*fD[i_fD]*tt[i_t]*nu[i_nu];
				REAL(i_fD,i_nu) += E_REAL(i_t,i_nu)*cos(phase) - E_IMAG(i_t,i_nu)*sin(phase);
				IMAG(i_fD,i_nu) += E_REAL(i_t,i_nu)*sin(phase) + E_IMAG(i_t,i_nu)*cos(phase);
			}
		}
	}
		
	#undef  E_REAL
	#undef  E_IMAG
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  Lambda (int N_t, int N_nu, int N_tau, double *t, double *L, double* tau, double* DS, double *hSS_real, double *hSS_im)
{
	//tell c++ how to read numpy arrays
	#define  DYNSPEC(i_t,i_nu)  DS[(i_t)*N_nu + (i_nu)]
	#define  REAL(i_t,i_tau)  hSS_real[(i_t)*N_tau + (i_tau)]
	#define  IMAG(i_t,i_tau)  hSS_im[(i_t)*N_tau + (i_tau)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double phase;
		for (int i_tau = 0; i_tau < N_tau; i_tau++){
			phase = -2.*M_PI*tau[i_tau]*L[i_nu];
			for (int i_t = 0; i_t < N_t; i_t++){
				REAL(i_t,i_tau) += DYNSPEC(i_t,i_nu)*cos(phase);
				IMAG(i_t,i_tau) += DYNSPEC(i_t,i_nu)*sin(phase);
			}
		}
	}
		
	#undef  DYNSPEC
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  SumStatPoints (int N_nu, int N_th, int N_t, double *nu, double *mu, double *ph, double *psi, double *wt, double *E_real, double *E_im)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double Phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_th = 0; i_th < N_th; i_th++){
				Phase = ph[i_th]+nu[i_nu]*pow(psi[i_th]+wt[i_t],2);
				REAL(i_t,i_nu) += mu[i_th]*cos(Phase);
				IMAG(i_t,i_nu) += mu[i_th]*sin(Phase);
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  SumStatPoints_GlobTVar (int N_nu, int N_th, int N_t, double *nu, double *mu, double *ph, double *psi, double *wt, double *TVar, double *E_real, double *E_im)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double Phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_th = 0; i_th < N_th; i_th++){
				Phase = ph[i_th]+nu[i_nu]*pow(psi[i_th]+wt[i_t],2);
				REAL(i_t,i_nu) += TVar[i_t]*mu[i_th]*cos(Phase);
				IMAG(i_t,i_nu) += TVar[i_t]*mu[i_th]*sin(Phase);
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  SumStatPoints_SingleTVar (int N_nu, int N_th, int N_t, double *nu, double *mu, double *ph, double *psi, double *wt, double *TVar, double *E_real, double *E_im)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	#define  TVAR(i_th,i_t)  TVar[(i_th)*N_t + (i_t)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double Phase;
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_th = 0; i_th < N_th; i_th++){
				Phase = ph[i_th]+nu[i_nu]*pow(psi[i_th]+wt[i_t],2);
				REAL(i_t,i_nu) += TVAR(i_th,i_t)*mu[i_th]*cos(Phase);
				IMAG(i_t,i_nu) += TVAR(i_th,i_t)*mu[i_th]*sin(Phase);
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
	#undef  TVAR
}

extern "C"
void  SumStatPoints_TDrift (int N_nu, int N_th, int N_t, int N_D, double slope, double *nu, double *mu, double *ph, double *psi, double *wt, double *psi_Drift, double *Drift, double *E_real, double *E_im)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_nu = 0; i_nu < N_nu; i_nu++){
		double Phase;
		int i_Drift;
		double width_Drift = psi_Drift[N_D-1]-psi_Drift[0];
		for (int i_t = 0; i_t < N_t; i_t++){
			for (int i_th = 0; i_th < N_th; i_th++){
				Phase = ph[i_th]+nu[i_nu]*pow(psi[i_th]+wt[i_t],2);
				i_Drift = (int) ( (psi[i_th]-psi_Drift[0]-slope*wt[i_t])/width_Drift*N_D + 0.5 );
				if (i_Drift>=0 && i_Drift<N_D) {
					REAL(i_t,i_nu) += Drift[i_Drift]*mu[i_th]*cos(Phase);
					IMAG(i_t,i_nu) += Drift[i_Drift]*mu[i_th]*sin(Phase);
				}
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  SumStatPoints_2scr (double *E_real, double *E_im, int N_t, int N_nu, double *t, double *nu, int N_x, int N_y, double *im_x, double *im_y, double D_x, double D_y, double D_s,
	double V_x, double V_y, double *V_p_ra, double *V_p_dec, double V_s_ra, double V_s_dec, double a_x, double a_y, double *mu_x, double *mu_y, double *phi_x, double *phi_y)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_t = 0; i_t < N_t; i_t++){
		double Phase;
		double Phi_opt;
		double vc = 299792458.;
		//double D_xy = D_y-D_x;
		double D_xs = D_s-D_x;
		double D_ys = D_s-D_y;
		double s = sin(a_x-a_y);
		double c = cos(a_x-a_y);
		double V_p_par = V_p_ra[i_t]*cos(a_x) + V_p_dec[i_t]*sin(a_x);
		double V_p_ort = -V_p_ra[i_t]*sin(a_x) + V_p_dec[i_t]*cos(a_x);
		double V_s_par = V_s_ra*cos(a_y) + V_s_dec*sin(a_y);
		double V_s_ort = -V_s_ra*sin(a_y) + V_s_dec*cos(a_y);
		
		double prefactor = 1./(D_y*D_xs-D_x*D_ys*c*c);
		double f_xx = prefactor*D_s*D_y/D_x;
		double f_xy = -prefactor*2.*c*D_s;
		double f_yy = prefactor*D_s*D_xs/D_ys;
		double f_xp_par = -2./D_x;
		double f_ys_par = -2./D_ys;
		double f_xp_ort = -prefactor*2.*s*c*D_ys;
		double f_xs_ort = -prefactor*2.*s*D_y;
		double f_yp_ort = prefactor*2.*s*D_xs;
		double f_ys_ort = prefactor*2.*s*c*D_x;
		
		double p_par = V_p_par*t[i_t];
		double p_ort = V_p_ort*t[i_t];
		double s_par = V_s_par*t[i_t];
		double s_ort = V_s_ort*t[i_t];
		double x;
		double y;
		
		for (int i_x = 0; i_x < N_x; i_x++){
			x = im_x[i_x] + V_x*t[i_t];
			for (int i_y = 0; i_y < N_y; i_y++){
				y = im_y[i_y] + V_y*t[i_t];
				Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
				for (int i_nu = 0; i_nu < N_nu; i_nu++){
					Phase = phi_x[i_x] + phi_y[i_y] + nu[i_nu]*Phi_opt;
					REAL(i_t,i_nu) += mu_x[i_x]*mu_y[i_y]*cos(Phase);
					IMAG(i_t,i_nu) += mu_x[i_x]*mu_y[i_y]*sin(Phase);
				}
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
}

extern "C"
void  SumStatPoints_2scr_CP (double *E_real, double *E_im, int N_t, int N_nu, double *t, double *nu, int N_x, int N_y, double *im_x, double *im_y, double D_x, double D_y, double D_s,
	double V_x, double V_y, double *V_p_ra, double *V_p_dec, double V_s_ra, double V_s_dec, double a_x, double a_y, double *mu_x, double *mu_y, double *phi_x, double *phi_y, double mu_CPx, double mu_CPy)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_t = 0; i_t < N_t; i_t++){
		double Phase;
		double Phi_opt;
		double vc = 299792458.;
		//double D_xy = D_y-D_x;
		double D_xs = D_s-D_x;
		double D_ys = D_s-D_y;
		double s = sin(a_x-a_y);
		double c = cos(a_x-a_y);
		double V_p_par = V_p_ra[i_t]*cos(a_x) + V_p_dec[i_t]*sin(a_x);
		double V_p_ort = -V_p_ra[i_t]*sin(a_x) + V_p_dec[i_t]*cos(a_x);
		double V_s_par = V_s_ra*cos(a_y) + V_s_dec*sin(a_y);
		double V_s_ort = -V_s_ra*sin(a_y) + V_s_dec*cos(a_y);
		
		double prefactor = 1./(D_y*D_xs-D_x*D_ys*c*c);
		double f_xx = prefactor*D_s*D_y/D_x;
		double f_xy = -prefactor*2.*c*D_s;
		double f_yy = prefactor*D_s*D_xs/D_ys;
		double f_xp_par = -2./D_x;
		double f_ys_par = -2./D_ys;
		double f_xp_ort = -prefactor*2.*s*c*D_ys;
		double f_xs_ort = -prefactor*2.*s*D_y;
		double f_yp_ort = prefactor*2.*s*D_xs;
		double f_ys_ort = prefactor*2.*s*c*D_x;
		
		double fx_p_par = (D_y*D_xs-D_x*D_ys*c*c)/(D_s*D_y);
		double fx_y_par = c*D_x/D_y;
		double fx_p_ort = s*c*(D_ys*D_x)/(D_s*D_y);
		double fx_s_ort = s*D_x/D_s;
		
		double fy_s_par = (D_y*D_xs-D_x*D_ys*c*c)/(D_s*D_xs);
		double fy_x_par = c*D_ys/D_xs;
		double fy_p_ort = -s*D_ys/D_s;
		double fy_s_ort = -s*c*D_x*D_ys/(D_s*D_xs);
		
		double p_par = V_p_par*t[i_t];
		double p_ort = V_p_ort*t[i_t];
		double s_par = V_s_par*t[i_t];
		double s_ort = V_s_ort*t[i_t];
		double x;
		double y;
		
		for (int i_x = 0; i_x < N_x; i_x++){
			x = im_x[i_x] + V_x*t[i_t];
			for (int i_y = 0; i_y < N_y; i_y++){
				y = im_y[i_y] + V_y*t[i_t];
				Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
				for (int i_nu = 0; i_nu < N_nu; i_nu++){
					Phase = phi_x[i_x] + phi_y[i_y] + nu[i_nu]*Phi_opt;
					REAL(i_t,i_nu) += mu_x[i_x]*mu_y[i_y]*cos(Phase);
					IMAG(i_t,i_nu) += mu_x[i_x]*mu_y[i_y]*sin(Phase);
				}
			}
		}
		for (int i_y = 0; i_y < N_y; i_y++){
			y = im_y[i_y] + V_y*t[i_t];
			x = fx_p_par*p_par + fx_y_par*y + fx_p_ort*p_ort + fx_s_ort*s_ort;
			Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
			for (int i_nu = 0; i_nu < N_nu; i_nu++){
				Phase = phi_y[i_y] + nu[i_nu]*Phi_opt;
				REAL(i_t,i_nu) += mu_CPx*mu_y[i_y]*cos(Phase);
				IMAG(i_t,i_nu) += mu_CPx*mu_y[i_y]*sin(Phase);
			}
		}
		for (int i_x = 0; i_x < N_x; i_x++){
			x = im_x[i_x] + V_x*t[i_t];
			y = fy_s_par*s_par + fy_x_par*x + fy_p_ort*p_ort + fy_s_ort*s_ort;
			Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
			for (int i_nu = 0; i_nu < N_nu; i_nu++){
				Phase = phi_x[i_x] + nu[i_nu]*Phi_opt;
				REAL(i_t,i_nu) += mu_CPy*mu_x[i_x]*cos(Phase);
				IMAG(i_t,i_nu) += mu_CPy*mu_x[i_x]*sin(Phase);
			}
		}
		x = D_xs/D_s*p_par + D_x/D_s*(c*s_par+s*s_ort);
		y = D_y/D_s*s_par + D_ys/D_s*(c*p_par-s*p_ort);
		Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
		for (int i_nu = 0; i_nu < N_nu; i_nu++){
			Phase = nu[i_nu]*Phi_opt;
			REAL(i_t,i_nu) += mu_CPx*mu_CPy*cos(Phase);
			IMAG(i_t,i_nu) += mu_CPx*mu_CPy*sin(Phase);
		}
	}
	
	#undef  REAL
	#undef  IMAG
}

/* extern "C"
void  SumStatPoints_CPx (double *E_real, double *E_im, int N_t, int N_nu, double *t, double *nu, int N_y, double *im_y, double D_x, double D_y, double D_s,
	double V_x, double V_y, double *V_p_ra, double *V_p_dec, double V_s_ra, double V_s_dec, double a_x, double a_y, double mu_x, double *mu_y, double *phi_y)
{
	#define  REAL(i_t,i_nu)  E_real[(i_t)*N_nu + (i_nu)]
	#define  IMAG(i_t,i_nu)  E_im[(i_t)*N_nu + (i_nu)]
	
	#pragma omp parallel for
	for (int i_t = 0; i_t < N_t; i_t++){
		double Phase;
		double Phi_opt;
		double vc = 299792458.;
		double D_xy = D_y-D_x;
		double D_xs = D_s-D_x;
		double D_ys = D_s-D_y;
		double s = sin(a_x-a_y);
		double c = cos(a_x-a_y);
		double V_p_par = V_p_ra[i_t]*cos(a_x) + V_p_dec[i_t]*sin(a_x);
		double V_p_ort = -V_p_ra[i_t]*sin(a_x) + V_p_dec[i_t]*cos(a_x);
		double V_s_par = V_s_ra*cos(a_y) + V_s_dec*sin(a_y);
		double V_s_ort = -V_s_ra*sin(a_y) + V_s_dec*cos(a_y);
		
		double prefactor = 1./(D_y*D_xs-D_x*D_ys*c*c);
		double f_xx = prefactor*D_s*D_y/D_x;
		double f_xy = -prefactor*2.*c*D_s;
		double f_yy = prefactor*D_s*D_xs/D_ys;
		double f_xp_par = -2./D_x;
		double f_ys_par = -2./D_ys;
		double f_xp_ort = -prefactor*2.*s*c*D_ys;
		double f_xs_ort = -prefactor*2.*s*D_y;
		double f_yp_ort = prefactor*2.*s*D_xs;
		double f_ys_ort = prefactor*2.*s*c*D_x;
		
		double fx_p_par = (D_y*D_xs-D_x*D_ys*c*c)/(D_s*D_y);
		double fx_y_par = c*D_x/D_y;
		double fx_p_ort = s*c*(D_ys*D_x)/(D_s*D_y);
		double fx_s_ort = s*D_x/D_s;
		
		double p_par = V_p_par*t[i_t];
		double p_ort = V_p_ort*t[i_t];
		double s_par = V_s_par*t[i_t];
		double s_ort = V_s_ort*t[i_t];
		double x;
		double y;
		
		for (int i_y = 0; i_y < N_y; i_y++){
			y = im_y[i_y] + V_y*t[i_t];
			x = fx_p_par*p_par + fx_y_par*y + fx_p_ort*p_ort + fx_s_ort*s_ort;
			Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
			for (int i_nu = 0; i_nu < N_nu; i_nu++){
				Phase = phi_y[i_y] + nu[i_nu]*Phi_opt;
				REAL(i_t,i_nu) += mu_x*mu_y[i_y]*cos(Phase);
				IMAG(i_t,i_nu) += mu_x*mu_y[i_y]*sin(Phase);
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
} */

/* extern "C"
void  SumStatPoints_at_2ndSCR (double *E_real, double *E_im, int N_t, int N_nu, double *t, double *nu, int N_x, int N_y, double *pos_x, double *im_y, double D_x, double D_y, double D_s,
	double V_x, double V_y, double V_s, double *mu_y, double *phi_y)
{
	#define  REAL(i_t,i_nu,i_x)  E_real[i_t*N_nu*N_x + i_nu*N_x + i_x]
	#define  IMAG(i_t,i_nu,i_x)  E_im[i_t*N_nu*N_x + i_nu*N_x + i_x]
	
	#pragma omp parallel for
	for (int i_t = 0; i_t < N_t; i_t++){
		double Phase;
		double Phi_opt;
		double vc = 299792458.;
		double D_xy = D_y-D_x;
		double D_xs = D_s-D_x;
		double D_ys = D_s-D_y;
		
		double f_xy = -2./D_xy;
		double f_yy = D_xs/(D_xy*D_ys);
		double f_ys = -2./D_ys;
		
		double s = V_s*t[i_t];
		double x;
		double y;
		
		for (int i_x = 0; i_x < N_x; i_x++){
			x = pos_x[i_x] + V_x*t[i_t];
			for (int i_y = 0; i_y < N_y; i_y++){
				y = im_y[i_y] + V_y*t[i_t];
				Phi_opt = M_PI/vc*(f_xy*x*y + f_yy*y*y + f_ys*y*s);
				for (int i_nu = 0; i_nu < N_nu; i_nu++){
					Phase = phi_y[i_y] + nu[i_nu]*Phi_opt;
					REAL(i_t,i_nu,i_x) += mu_y[i_y]*cos(Phase);
					IMAG(i_t,i_nu,i_x) += mu_y[i_y]*sin(Phase);
				}
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
} */

extern "C"
void  SumStatPoints_at_2ndSCR (double *E_real, double *E_im, int N_t, int N_nu, double *t, double *nu, int N_x, int N_y, double *im_x, double *im_y, double D_x, double D_y, double D_s,
	double V_x, double V_y, double *V_p_ra, double *V_p_dec, double V_s_ra, double V_s_dec, double a_x, double a_y, double *mu_y, double *phi_y)
{
	#define  REAL(i_t,i_nu,i_x)  E_real[i_t*N_nu*N_x + i_nu*N_x + i_x]
	#define  IMAG(i_t,i_nu,i_x)  E_im[i_t*N_nu*N_x + i_nu*N_x + i_x]
	
	#pragma omp parallel for
	for (int i_t = 0; i_t < N_t; i_t++){
		double Phase;
		double Phi_opt;
		double vc = 299792458.;
		double D_xy = D_y-D_x;
		double D_xs = D_s-D_x;
		double D_ys = D_s-D_y;
		double s = sin(a_x-a_y);
		double c = cos(a_x-a_y);
		double V_p_par = V_p_ra[i_t]*cos(a_x) + V_p_dec[i_t]*sin(a_x);
		double V_p_ort = -V_p_ra[i_t]*sin(a_x) + V_p_dec[i_t]*cos(a_x);
		double V_s_par = V_s_ra*cos(a_y) + V_s_dec*sin(a_y);
		double V_s_ort = -V_s_ra*sin(a_y) + V_s_dec*cos(a_y);
		
		double prefactor = 1./(D_y*D_xs-D_x*D_ys*c*c);
		double f_xx = prefactor*D_s*D_y/D_x;
		double f_xy = -prefactor*2.*c*D_s;
		double f_yy = prefactor*D_s*D_xs/D_xy;
		double f_xp_par = -2./D_x;
		double f_ys_par = -2./D_ys;
		double f_xp_ort = -prefactor*2.*s*c*D_ys;
		double f_xs_ort = -prefactor*2.*s*D_y;
		double f_yp_ort = prefactor*2.*s*D_xs;
		double f_ys_ort = prefactor*2.*s*c*D_x;
		
		double p_par = V_p_par*t[i_t];
		double p_ort = V_p_ort*t[i_t];
		double s_par = V_s_par*t[i_t];
		double s_ort = V_s_ort*t[i_t];
		double x;
		double y;
		
		for (int i_x = 0; i_x < N_x; i_x++){
			x = im_x[i_x] + V_x*t[i_t];
			for (int i_y = 0; i_y < N_y; i_y++){
				y = im_y[i_y] + V_y*t[i_t];
				Phi_opt = M_PI/vc*(f_xx*x*x + f_xy*x*y + f_yy*y*y + f_xp_par*x*p_par + f_ys_par*y*s_par + f_xp_ort*x*p_ort + f_xs_ort*x*s_ort + f_yp_ort*y*p_ort + f_ys_ort*y*s_ort);
				for (int i_nu = 0; i_nu < N_nu; i_nu++){
					Phase = phi_y[i_y] + nu[i_nu]*Phi_opt;
					REAL(i_t,i_nu,i_x) += mu_y[i_y]*cos(Phase);
					IMAG(i_t,i_nu,i_x) += mu_y[i_y]*sin(Phase);
				}
			}
		}
	}
	
	#undef  REAL
	#undef  IMAG
}