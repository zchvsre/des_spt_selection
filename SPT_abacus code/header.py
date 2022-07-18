#lambda scaling relation from To et al. 2021 and To & Krause
import numpy as np
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