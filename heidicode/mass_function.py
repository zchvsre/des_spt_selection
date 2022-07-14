import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d


class MassFunction(object):
    """Read and process abacus mass function.

    Args:
        object (_type_): _description_
    """
    def __init__(self):
#         project_path = "/global/cfs/cdirs/des/zhou/spt_selection/"
#         halo_fname = 'data/abacus_mf.npy'
#         mass = np.load(os.path.join(project_path, halo_fname))

#         hist_data = plt.hist(np.log(mass), bins=50)
#         plt.title("Histogram of Log Mass Function")
#         plt.axvline(np.log(1E13))
#         plt.show()

#         n = np.log(hist_data[0])  #numbers in each bin
#         bins = hist_data[1]
#         bins = 0.5 * (bins[0:-1] + bins[1:])

#         bin_mid = []
#         slope_mid = []

#         for i in range(len(bins) - 1):
#             bin_mid.append(0.5 * (bins[i + 1] + bins[i]))
#             slope = (n[i + 1] - n[i]) / (bins[i + 1] - bins[i])
#             slope_mid.append(-slope)

        # bin_mid.append(np.log(1e16))
        # slope_mid.append(0)

#         mf_slope_interp = interp1d(bin_mid, slope_mid)
#         plt.plot(bin_mid, mf_slope_interp(bin_mid))
#         plt.title("Beta")
#         plt.show()

#         self.beta = mf_slope_interp
#         self.mass = mass[mass > 1E13]
        
        from scipy.stats import expon
        beta=1.6
        lnM = expon.rvs(loc=np.log(1e12), scale=1/beta, size=10**8)
        # lnM = lnM[lnM > np.log(1e13)]
        
        def beta(mu):
            return 1.6
        
        
        self.beta = beta
        self.mass = np.exp(lnM)
