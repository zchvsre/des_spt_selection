#Script passed on from A. Salcedo
import numpy as np
import scipy.integrate as integrate

def wp(rp,binmin,binmax,xi,pimax=None):
    """compute wp(r_p) from tabulated xi(r)."""

    lower_bound = rp
    upper_bound = np.sqrt(rp**2 + pimax**2)
    binmask = np.logical_and(binmax > lower_bound, binmin < upper_bound)
    masked_xi = xi[binmask]
    r_i = binmin[binmask]
    r_iplus = binmax[binmask]
    s_plus = np.minimum(upper_bound, r_iplus)
    s_minus = np.maximum(lower_bound, r_i)
    # here we assume that xi is piecewise constant over the tabulated input bins
    elementwise_integral = 2.0*masked_xi * \
                           (np.sqrt(s_plus**2 - rp**2) - np.sqrt(s_minus**2 - rp**2))
    w_p = np.sum(elementwise_integral)

    return w_p

def DeltaSigma(binmin,binmax,xi,pimax=None,mean_rho=None,H0=None, rp_min=None, rp_max=None, nbins=None):
    # mean rho (in comoving Msun pc^-2, no little h)
    h = H0/100.

    # compute rp bins
    #nbins = 40
    #rp_min = 0.1
    #rp_max = 125.0
    rp_bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins+1)
    rp_binmin = rp_bins[0:-1]
    rp_binmax = rp_bins[1:]
    rp_mid = (rp_binmin + rp_binmax)/2.0

    ds = np.zeros(rp_mid.shape[0])
    integrand = lambda r: r*wp(r,binmin,binmax,xi,pimax=pimax)
    for i in range(rp_mid.shape[0]):
        integral, abserr = integrate.quad(integrand, 0., rp_mid[i], epsabs=1.0e-2, epsrel=1.0e-2)
        ds[i] = (integral * (2.0/rp_mid[i]**2) - wp(rp_mid[i],binmin,binmax,xi,pimax=pimax)) * mean_rho

    # convert Mpc/h unit to pc (no h)
    #ds *= 1.0e6 / h
    ds *= 1.0e6 / h**2.0 #delta sigma comes out in Msun h / pc^2 units
    return rp_binmin, rp_binmax, ds

def DeltaSigma_from_files(header_file,filename,output_file,pimax,rp_min,rp_max,nbins):
    # read in cosmological parameters from header_file
    import config
    cf = config.AbacusConfigFile(header_file)
    omega_m = cf.Omega_M # at z=0
    H_0 = cf.H0
    
    # compute mean_rho (comoving density units = Msun pc^-3)
    speed_of_light_km_s = 2.998e5 # km/s
    csq_over_G = 2.494e12 # 3c^2/(8*pi*G) Msun pc^-1
    mean_rho = omega_m * csq_over_G * (H_0/speed_of_light_km_s)**2 / 1.0e12 # Msun pc^-3

    binmin,binmax,null,xi = np.loadtxt(filename,unpack=True)
    DS_binmin, DS_binmax, DS = DeltaSigma(binmin,binmax,xi,pimax=float(pimax),mean_rho=mean_rho,H0=H_0,rp_min=float(rp_min),rp_max=float(rp_max),nbins=int(nbins))
    np.savetxt(output_file, np.c_[DS_binmin, DS_binmax, np.zeros(DS.shape[0]), DS],
               delimiter='\t')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('header_file')
    parser.add_argument('output_file')
    parser.add_argument('pimax')
    parser.add_argument('rp_min')
    parser.add_argument('rp_max')
    parser.add_argument('nbins')
    args = parser.parse_args()

DeltaSigma_from_files(args.header_file, args.input_file, args.output_file, args.pimax, args.rp_min, args.rp_max, args.nbins)
