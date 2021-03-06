import numpy as np
from colossus.cosmology import cosmology

from kllr import kllr_model


class ScalingRelation(object):
    """Scaling relations we would like to use

    Args:
        object (_type_): _description_
    """
    def __init__(self,mass_vector=None, lam_vector=None):
        h = 0.6726
        params = {
            'flat': True,
            'H0': 67.26,
            'Om0': 0.14212 / h**2,
            'Ob0': 0.02222 / h**2,
            'sigma8': 0.81,
            'ns': 9.9652,
            'w0': -1.0,
            'Neff': 3.04
        }
        cosmo = cosmology.setCosmology('Abacus', params)
        
        h_70 = cosmo.H0 / 70
        h_100 = cosmo.H0 / 100
        
        if mass_vector == None:

            self.Mpiv_lam = 5E14
            self.alpha_lam = 0.939
            self.pi_lam = 4.25 - self.alpha_lam * np.log(
                self.Mpiv_lam) + 0.15 * np.log(1.3 / 1.45)
            self.scatter_lam = 0.36
            
        else:
            
            print("Fitting lam scaling relation with KLLR.")
            
           zero_lam_mask = (lam != 0)

            x = np.log(mass[zero_lam_mask])
            y = np.log(lam[zero_lam_mask])

            from kllr import kllr_model #https://github.com/afarahi/kllr
            lm = kllr_model(kernel_type = 'gaussian', kernel_width = 0.2)
            bins = 20
            x_bin, yrange_mean, intercept, slope, scatter, skew, kurt = lm.fit(x, y, bins=bins) 

            self.alpha_lam = np.average(slope)
            self.pi_lam = np.average(intercept)
            self.scatter_lam = np.average(scatter)

        print(f"{self.alpha_lam=}, {self.pi_lam=}, {self.scatter_lam=}")

        #SZ scaling relation from Bocquet et al. 2019
        self.Mpiv_SZ = 4.3E14 / 0.7
        self.alpha_SZ = 1.519
        self.pi_SZ = np.log(5.68) - 1.519 * np.log(
            self.Mpiv_SZ) + 0.547 * np.log(cosmo.Ez(0.3) / cosmo.Ez(0.6))
        self.scatter_SZ = 0.152

        print(f"{self.alpha_SZ=}, {self.pi_SZ=}, {self.scatter_SZ=}")

        #Mwl scaling relation

        self.alpha_Mwl = 1
        self.pi_Mwl = 0
        self.scatter_Mwl = 0.5

        print(f"{self.alpha_Mwl=}, {self.pi_Mwl=}, {self.scatter_Mwl=}")