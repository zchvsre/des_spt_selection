import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.stats import expon


class MassFunction(object):
    """Read and process abacus mass function.

    Args:
        object (_type_): _description_
    """
    def __init__(self, kind):

        beta_ = 1.6

        if kind == "exponential":
            lnM = expon.rvs(loc=np.log(1e12), scale=1 / beta_, size=10**8)

            def beta(mu):

                if type(mu) == np.float64:
                    return beta_
                else:
                    return beta_ * np.ones(len(mu))

            self.beta = beta
            self.mass = np.exp(lnM)

        elif kind == "nbody":
            project_path = "/global/cfs/cdirs/des/zhou/spt_selection/"
            halo_fname = 'data/abacus_mf.npy'
            mass = np.load(os.path.join(project_path, halo_fname))

            hist_data = np.histogram(np.log(mass), bins=30)

            n = np.log(hist_data[0])  #numbers in each bin
            bins = hist_data[1]
            bins = 0.5 * (bins[0:-1] + bins[1:])

            bin_mid = []
            slope_mid = []

            for i in range(len(bins) - 1):
                bin_mid.append(0.5 * (bins[i + 1] + bins[i]))
                slope = (n[i + 1] - n[i]) / (bins[i + 1] - bins[i])
                slope_mid.append(-slope)

            slope_mid = np.array(slope_mid)
            slope_mid[slope_mid > 1.5] = 1

            mf_slope_interp = interp1d(bin_mid,
                                       slope_mid,
                                       bounds_error=False,
                                       fill_value="extrapolate")

            self.beta = mf_slope_interp
            self.mass = mass

        #   project_path = "/global/cfs/cdirs/des/zhou/spt_selection/"


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

        plt.hist(np.log(self.mass), label=f"N={len(self.mass)}", bins=30)
        plt.title("Histogram of Log Mass")
        plt.legend()
        plt.show()

        x_for_beta = np.linspace(np.log(1E13), np.log(self.mass.max()), 20)
        plt.plot(x_for_beta, self.beta(x_for_beta))
        plt.title("Beta for Mass")
