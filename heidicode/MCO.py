import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_histogram
from scipy.interpolate import interp1d
from scipy.integrate import simps
from colossus.cosmology import cosmology


h = 0.6726
params = {'flat' : True, 'H0' : 67.26,  'Om0' : 0.14212/h**2, 'Ob0' : 0.02222/h**2, 'sigma8' : 0.81, 'ns' : 9.9652, 'w0' : -1.0, 'Neff' : 3.04}
cosmo = cosmology.setCosmology('Abacus',params)
h_70 = cosmo.H0/70
h_100 = cosmo.H0/100

Mpiv_lam = 5E14
alpha_lam = 0.939
pi_lam = 4.25 - alpha_lam*np.log(Mpiv_lam) + 0.15*np.log(1.3/1.45)
scatter_lam = 0.36

#SZ scaling relation from Bocquet et al. 2019
Mpiv_SZ = 4.3E14/0.7
alpha_SZ = 1.519
pi_SZ = np.log(5.68) - 1.519*np.log(Mpiv_SZ) + 0.547 * np.log(cosmo.Ez(0.3)/cosmo.Ez(0.6))
scatter_SZ = 0.152

#Mwl scaling relation
 
alpha_Mwl = 1
pi_Mwl = 0
scatter_Mwl = 0.5


class MonteCarloObservables(object):
    
    def __init__(self, nh, r, lnM,
                 lnlam_mean=np.nan, lnSZ_mean = np.nan, lnMwl_mean = np.nan,
                 scatter_Mwl=np.nan, scatter_lam=np.nan, scatter_SZ=np.nan,
                 mf_slope_interp=None):    
    
        self.r = r
        self.scatter_Mwl = scatter_Mwl
        self.scatter_lam = scatter_lam
        self.scatter_SZ = scatter_SZ
        
        mv = multivariate_normal([0, 0], [[1, r], [r, 1]])
        rv = mv.rvs(size=nh)
        x = rv[:,0]
        y = rv[:,1] 
        
        gauss = norm(0,1)
        z = gauss.rvs(size=nh)
 
        self.lnlam = lnlam_mean + scatter_lam * x
        self.lnMwl = lnMwl_mean + scatter_Mwl * y
        self.lnSZ = lnSZ_mean + scatter_SZ * z 
        
        self.mf_slope_interp = mf_slope_interp
        
        self.P_lam = rv_histogram(np.histogram(self.lnlam,bins=100))
        self.P_SZ = rv_histogram(np.histogram(self.lnSZ,bins=100))
            
    
    def TH_calculate_mean_Mwl_given_lam_SZ(self,lam,SZ,correction=True):
                
        mu_lam = (lam-pi_lam)/alpha_lam
        mu_SZ = (SZ-pi_SZ)/alpha_SZ
        
        sig_lam = scatter_lam/alpha_lam
        sig_SZ = scatter_SZ/alpha_SZ
    
        mu_guess = (mu_lam/(sig_lam)**2 + mu_SZ/(sig_SZ)**2) / (1/sig_lam**2 + 1/sig_SZ**2)

        if correction == True:
            beta = self.mf_slope_interp(mu_guess)
            beta[mu_guess < 30.3] = 1.06
        else:
            beta = np.zeros(np.shape(mu_guess))
            self.r = 0
        
        # print("mu_given_SZ_lam, beta:", mu_guess, beta)
        
        
        mu_given_lam_SZ_num = (alpha_lam/self.scatter_lam**2)*(lam-pi_lam) + (alpha_SZ/self.scatter_SZ**2)*(SZ-pi_SZ) - beta
        mu_given_lam_SZ_den = (alpha_lam/self.scatter_lam)**2 + (alpha_SZ/self.scatter_SZ)**2
        mu_given_lam_SZ = mu_given_lam_SZ_num/mu_given_lam_SZ_den

        mu_given_lam = (lam - pi_lam)/alpha_lam
        mu_given_SZ = (SZ - pi_SZ)/alpha_SZ

        third_term_num = self.r * self.scatter_Mwl * (self.scatter_lam/alpha_lam) * (mu_given_lam - mu_given_SZ + beta*(self.scatter_SZ/alpha_SZ)**2)
        third_term_den = (self.scatter_SZ/alpha_SZ)**2 + (self.scatter_lam/alpha_lam)**2

        third_term = third_term_num/third_term_den

        TH_Mwl_given_lambda_SZ = pi_Mwl + alpha_Mwl*mu_given_lam_SZ + third_term
        # TH_Mwl_given_lambda_SZ = third_term

        return (TH_Mwl_given_lambda_SZ)
    
    
    def mean_Mwl_in_bin(self, lam1, lam2, SZ1, SZ2):
        
        norm_factor_lam = self.P_lam.cdf(lam2) - self.P_lam.cdf(lam1)
        norm_factor_SZ = self.P_SZ.cdf(SZ2) - self.P_SZ.cdf(SZ1)
                
        lam_range = np.linspace(lam1,lam2,10000)
        SZ_range = np.linspace(SZ1,SZ2,10000)
        
        
        mesh = self.TH_calculate_mean_Mwl_given_lam_SZ(lam_range.reshape(-1,1),SZ_range.reshape(1,-1),correction=True) * self.P_lam.pdf(lam_range.reshape(-1,1))/norm_factor_lam * self.P_SZ.pdf(SZ_range.reshape(1,-1))/norm_factor_SZ
                
        integral = simps([simps(SZ,SZ_range) for SZ in mesh],lam_range)
        
        return integral


    def MC_calculate_mean_Mwl_given_lam_SZ(self, nbins, correction):
        
        # lnlam_bins = pd.qcut(self.lnlam,nbins,retbins=True)[1]
        # lnSZ_bins = pd.qcut(self.lnSZ,nbins,retbins=True)[1]
        
        lnlam_bins = np.log(np.array([20,40,45,50]))
        lnSZ_bins = np.log(np.array([0.1,1,2,20]))

                
        diff_array = np.empty([nbins-1])
        SZ_array = np.empty([nbins-1])
        lam_array = np.empty([nbins-1])
        
        diff_array = np.zeros([nbins-1,nbins-1])
        count_array = np.zeros([nbins-1, nbins-1])
        
        for i in range(nbins-1): # go over each SZ bin
            for j in range(nbins-1): #go over each lamdba bin

                SZ_left_edge, SZ_right_edge = lnSZ_bins[i],lnSZ_bins[i+1]
                lam_left_edge, lam_right_edge = lnlam_bins[j],lnlam_bins[j+1]
                
                SZ_mid = (SZ_left_edge+SZ_right_edge)/2.
                lam_mid = (lam_left_edge+lam_right_edge)/2.
                
                SZ_array[i] = SZ_mid
                lam_array[j] = lam_mid

                SZ_mask = (self.lnSZ > SZ_left_edge) & (self.lnSZ <= SZ_right_edge)
                lam_mask = (self.lnlam > lam_left_edge) & (self.lnlam <= lam_right_edge)
                
                SZ_median = np.median(self.lnSZ[SZ_mask])
                lam_median = np.median(self.lnlam[lam_mask])

                total_mask = SZ_mask & lam_mask  #combine the richness and SZ mask
                count_array[i][j] = np.sum(total_mask)

                if np.sum(total_mask) != 0:
                    
                    TH_Mwl_given_lam_SZ = self.mean_Mwl_in_bin(lam_left_edge, lam_right_edge, SZ_left_edge,SZ_right_edge)

                    diff = (TH_Mwl_given_lam_SZ - np.mean(self.lnMwl[total_mask]))

                    diff_array[i][j] = diff
                    
                else:
                    diff_array[i][j] = 0

                # print("----------------------------------------------------------")

            
        return(lam_array,SZ_array,diff_array,count_array)
            
            #plot x axis r, bin by SZ and lambda
            #plot x axis lambda, bin by SZ
            #y axis difference