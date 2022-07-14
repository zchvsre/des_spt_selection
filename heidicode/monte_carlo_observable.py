import numpy as np
import pandas as pd
import scipy as sp
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_histogram
from scipy.interpolate import interp1d
from scipy.integrate import romb, simps
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
import scipy.signal

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

Mpiv_lam = 5E14
alpha_lam = 0.939
pi_lam = 4.25 - alpha_lam * np.log(Mpiv_lam) + 0.15 * np.log(1.3 / 1.45)
scatter_lam = 0.36

#SZ scaling relation from Bocquet et al. 2019
Mpiv_SZ = 4.3E14 / 0.7
alpha_SZ = 1.519
pi_SZ = np.log(5.68) - 1.519 * np.log(Mpiv_SZ) + 0.547 * np.log(
    cosmo.Ez(0.3) / cosmo.Ez(0.6))
scatter_SZ = 0.152
#Mwl scaling relation

alpha_Mwl = 1
pi_Mwl = 0
scatter_Mwl = 0.1


class MonteCarloObservables(object):
    """Monte Carlo Observables

    Args:
        object (_type_): _description_
    """
    def __init__(self,
                 nh,
                 r,
                 lnM,
                 lnlam_mean=np.nan,
                 lnSZ_mean=np.nan,
                 lnMwl_mean=np.nan,
                 scatter_Mwl=np.nan,
                 scatter_lam=np.nan,
                 scatter_SZ=np.nan,
                 mf_slope_interp=None):

        self.r = r
        self.scatter_Mwl = scatter_Mwl
        self.scatter_lam = scatter_lam
        self.scatter_SZ = scatter_SZ

        mv = multivariate_normal([0, 0], [[1, r], [r, 1]])
        rv = mv.rvs(size=nh)
        x = rv[:, 0]
        y = rv[:, 1]

        gauss = norm(0, 1)
        z = gauss.rvs(size=nh)

        self.lnlam = lnlam_mean + scatter_lam * x
        self.lnMwl = lnMwl_mean + scatter_Mwl * y
        self.lnSZ = lnSZ_mean + scatter_SZ * z

        self.mf_slope_interp = mf_slope_interp

        multiplier = 1000
        rv = mv.rvs(size=nh * multiplier)
        x = rv[:, 0]
        y = rv[:, 1]

        gauss = norm(0, 1)
        z = gauss.rvs(size=nh * multiplier)

        self.lnlam_for_pdf = np.repeat(lnlam_mean,
                                       multiplier) + scatter_lam * x
        self.lnMwl_for_pdf = np.repeat(lnMwl_mean,
                                       multiplier) + scatter_Mwl * y
        self.lnSZ_for_pdf = np.repeat(lnSZ_mean, multiplier) + scatter_SZ * z

        # self.P_lam = rv_histogram(
        #     np.histogram(self.lnlam,
        #                  bins=np.linspace(np.log(1), np.log(50), 30)))
        # self.P_SZ = rv_histogram(
        #     np.histogram(self.lnSZ,
        #                  bins=np.linspace(np.log(0.001), np.log(5), 100)))

        # self.P_lam = rv_histogram(np.histogram(self.lnlam, bins=100))
        # self.P_SZ = rv_histogram(np.histogram(self.lnSZ, bins=100))

    def theory_calculate_mean_mwl_given_lam_sz(self, lam, SZ, correction=True):
        """Calculate the mean lensing mass given lambda and SZ

        Args:
            lam (_type_): _description_
            SZ (_type_): _description_
            correction (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        mu_lam = (lam - pi_lam) / alpha_lam
        mu_SZ = (SZ - pi_SZ) / alpha_SZ

        sig_lam = scatter_lam / alpha_lam
        sig_SZ = scatter_SZ / alpha_SZ

        mu_guess = (mu_lam / (sig_lam)**2 + mu_SZ /
                    (sig_SZ)**2) / (1 / sig_lam**2 + 1 / sig_SZ**2)

        beta = self.mf_slope_interp(mu_guess)

        mu_given_lam_SZ_num = (alpha_lam / self.scatter_lam**2) * (
            lam - pi_lam) + (alpha_SZ / self.scatter_SZ**2) * (SZ -
                                                               pi_SZ) - beta
        mu_given_lam_SZ_den = (alpha_lam / self.scatter_lam)**2 + (
            alpha_SZ / self.scatter_SZ)**2
        mu_given_lam_SZ = mu_given_lam_SZ_num / mu_given_lam_SZ_den

        mu_given_lam = (lam - pi_lam) / alpha_lam
        mu_given_SZ = (SZ - pi_SZ) / alpha_SZ

        if correction is True:

            third_term_num = self.r * self.scatter_Mwl * (
                self.scatter_lam /
                alpha_lam) * (mu_given_lam - mu_given_SZ + beta *
                              (self.scatter_SZ / alpha_SZ)**2)
            third_term_den = (self.scatter_SZ /
                              alpha_SZ)**2 + (self.scatter_lam / alpha_lam)**2
            third_term = third_term_num / third_term_den
        else:
            third_term = 0

        theory_mwl_given_lambda_SZ = pi_Mwl + alpha_Mwl * mu_given_lam_SZ + third_term

        return (theory_mwl_given_lambda_SZ)

    def mean_mwl_in_bin(self, lam1, lam2, sz1, sz2, correction):
        """Calculate the precise lensing mass given lambda and SZ bin

        Args:
            lam1 (_type_): _description_
            lam2 (_type_): _description_
            SZ1 (_type_): _description_
            SZ2 (_type_): _description_
            correction (_type_): _description_

        Returns:
            _type_: _description_
        """

        # norm_factor_lam = self.P_lam.cdf(lam2) - self.P_lam.cdf(lam1)
        # norm_factor_SZ = self.P_SZ.cdf(SZ2) - self.P_SZ.cdf(SZ1)

        # norm_factor_lam = self.lam_kde.integrate_box_1d(lam1, lam2)
        # norm_factor_sz = self.sz_kde.integrate_box_1d(sz1, sz2)

        norm_factor_lam = 1
        norm_factor_sz = 1

        print("The normalization factors are:", norm_factor_lam,
              norm_factor_sz)

        lam_range, lam_step = np.linspace(lam1, lam2, NSTEPS, retstep=True)
        sz_range, sz_step = np.linspace(sz1, sz2, NSTEPS, retstep=True)

        # lam_p = self.lam_kde.pdf(lam_range)
        # sz_p = self.sz_kde.pdf(sz_range)

        lam_p = self.lam_pdf(lam_range)
        sz_p = self.sz_pdf(sz_range)

        print("The number of out of bound points are", np.sum(lam_p == 0),
              np.sum(sz_p == 0))
        lam_p[0] = (lam_p[1] - lam_p[2]) + lam_p[1]
        lam_p[-1] = (lam_p[-2] - lam_p[-3]) + lam_p[-2]
        sz_p[0] = (sz_p[1] - sz_p[2]) + sz_p[1]
        sz_p[-1] = (sz_p[-2] - sz_p[-3]) + sz_p[-2]
        print("The number of out of bound points are", np.sum(lam_p == 0),
              np.sum(sz_p == 0))

        plt.plot(lam_range, lam_p)
        plt.title("lam PDF to be put in the integral")
        plt.show()

        plt.plot(sz_range, sz_p)
        plt.title("sz PDF to be put in the integral")
        plt.show()
        # lam_p_smooth = scipy.signal.savgol_filter(lam_p, 5, 1)
        # sz_p_smooth = scipy.signal.savgol_filter(sz_p, 5, 1)

        # plt.plot(lam_range, lam_p_smooth)
        # plt.show()
        # plt.plot(sz_range, sz_p_smooth)
        # plt.show()

        integral = 0

        for i, lam in tqdm(enumerate(lam_range)):
            for j, sz in enumerate(sz_range):
                # print(lam, sz)
                p_lam = lam_step * lam_p[i]
                p_sz = sz_step * sz_p[j]
                mean_mwl = self.theory_calculate_mean_mwl_given_lam_sz(
                    lam, sz, correction=correction)
                integral += p_lam * p_sz * mean_mwl

        integral /= norm_factor_lam
        integral /= norm_factor_sz

        # mesh = self.theory_calculate_mean_mwl_given_lam_sz(
        #     lam_range.reshape(-1, 1),
        #     SZ_range.reshape(1, -1),
        #     correction=correction) * self.P_lam(
        #         lam_range.reshape(-1, 1)) / norm_factor_lam * self.P_SZ(
        #             SZ_range.reshape(1, -1)) / norm_factor_SZ

        # mesh = self.theory_calculate_mean_mwl_given_lam_sz(
        #     lam_range.reshape(-1, 1),
        #     SZ_range.reshape(1, -1),
        #     correction=correction) * self.P_lam.pdf(lam_range.reshape(
        #         -1, 1)) * self.P_SZ.pdf(SZ_range.reshape(1, -1))

        # mesh_norm = self.P_lam.pdf(lam_range.reshape(-1, 1)) * self.P_SZ.pdf(
        #     SZ_range.reshape(1, -1))

        # integral = romb([romb(SZ, SZ_step) for SZ in mesh], lam_step)
        # integral = simps([simps(SZ, SZ_range) for SZ in mesh], lam_range)

        return integral

    def mc_calculate_mean_mwl_given_lam_sz(self, nbins, correction, lam1, lam2,
                                           sz1, sz2, NSTEPS):
        """Calculate the mean lensing mass given lambda and SZ in Monte Carlo.

        Args:
            nbins (_type_): _description_
            correction (_type_): _description_

        Returns:
            _type_: _description_
        """

        lnlam_bins = np.log(np.array([lam1, lam2]))
        lnSZ_bins = np.log(np.array([sz1, sz2]))

        #pdf from scipy kde
        # def get_pdf_in_bin(data, left_edge, right_edge):
        #     data_in_bin = np.ma.masked_outside(
        #         data, left_edge, right_edge).compressed()[:, np.newaxis]
        #     x_values = np.linspace(np.min(data_in_bin),
        #                            np.max(data_in_bin))[:, np.newaxis]
        #     kde = KernelDensity(kernel="gaussian",
        #                         bandwidth=0.001).fit(data_in_bin)
        #     log_den = kde.score_samples(x_values)

        #     def pdf(x_values):
        #         x_values = x_values[:, np.newaixs]
        #         return (np.exp(kde.score_samples(x_values)))

        #pdf from rv_histogram

        def get_pdf_in_bin(data, left_edge, right_edge):

            data_in_bin = np.ma.masked_outside(data, left_edge,
                                               right_edge).compressed()
            rv = sp.stats.rv_histogram(np.histogram(data_in_bin, bins=500))
            plt.hist(data_in_bin)
            plt.show()
            plt.plot(
                np.linspace(data_in_bin.min(), data_in_bin.max(), 10000),
                rv.pdf(np.linspace(data_in_bin.min(), data_in_bin.max(),
                                   10000)))
            plt.show()
            # return (rv.pdf)
            return (rv.pdf)

        #pdf from interpolating histogram
        # def get_pdf_in_bin(data, left_edge, right_edge):
        #     data_in_bin = np.ma.masked_outside(data, left_edge,
        #                                        right_edge).compressed()
        #     counts, bin_edges = np.histogram(data_in_bin,
        #                                      density=True,
        #                                      bins=10)
        #     bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        #     pdf = interp1d(bin_mid,
        #                    counts,
        #                    bounds_error=False,
        #                    fill_value="extrapolate")
        #     plt.hist(data_in_bin, bins=20)
        #     plt.show()
        #     plt.plot(np.linspace(left_edge, right_edge, 1000),
        #              pdf(np.linspace(left_edge, right_edge, 1000)))
        #     plt.show()
        #     return (pdf)

        self.lam_pdf = get_pdf_in_bin(self.lnlam_for_pdf, np.log(lam1),
                                      np.log(lam2))
        self.sz_pdf = get_pdf_in_bin(self.lnSZ_for_pdf, np.log(sz1),
                                     np.log(sz2))

        #pdf from kde estimate
        # def get_pdf_in_bin(data, left_edge, right_edge):
        #     data_in_bin = np.ma.masked_outside(data, left_edge - 0.1,
        #                                        right_edge + 0.1).compressed()

        #     kde = sp.stats.gaussian_kde(data_in_bin)

        #     plt.hist(data_in_bin)
        #     plt.show()
        #     plt.plot(np.linspace(left_edge, right_edge, 1000),
        #              kde.pdf(np.linspace(left_edge, right_edge, 1000)))
        #     plt.show()
        #     return (kde)

        # self.lam_kde = get_pdf_in_bin(self.lnlam, np.log(lam1), np.log(lam2))
        # self.sz_kde = get_pdf_in_bin(self.lnSZ, np.log(sz1), np.log(sz2))

        # lnlam_bins = pd.qcut(self.lnlam,nbins,retbins=True)[1]
        # lnSZ_bins = pd.qcut(self.lnSZ,nbins,retbins=True)[1]

        diff_array = np.empty([nbins - 1])
        SZ_array = np.empty([nbins - 1])
        lam_array = np.empty([nbins - 1])

        diff_array = np.zeros([nbins - 1, nbins - 1])
        count_array = np.zeros([nbins - 1, nbins - 1])

        for i in range(nbins - 1):  # go over each SZ bin
            for j in range(nbins - 1):  #go over each lamdba bin

                SZ_left_edge, SZ_right_edge = lnSZ_bins[i], lnSZ_bins[i + 1]
                lam_left_edge, lam_right_edge = lnlam_bins[j], lnlam_bins[j +
                                                                          1]

                SZ_mid = (SZ_left_edge + SZ_right_edge) / 2.
                lam_mid = (lam_left_edge + lam_right_edge) / 2.

                SZ_array[i] = SZ_mid
                lam_array[j] = lam_mid

                SZ_mask = (self.lnSZ > SZ_left_edge) & (self.lnSZ <=
                                                        SZ_right_edge)
                lam_mask = (self.lnlam > lam_left_edge) & (self.lnlam <=
                                                           lam_right_edge)

                SZ_median = np.median(self.lnSZ[SZ_mask])
                lam_median = np.median(self.lnlam[lam_mask])

                total_mask = SZ_mask & lam_mask  #combine the richness and SZ mask
                count_array[i][j] = np.sum(total_mask)

                print("Lam bounds are", np.exp(lam_left_edge),
                      np.exp(lam_right_edge))
                print("SZ bounds are", np.exp(SZ_left_edge),
                      np.exp(SZ_right_edge))

                theory_mwl_given_lam_sz = self.mean_mwl_in_bin(
                    lam_left_edge, lam_right_edge, SZ_left_edge, SZ_right_edge,
                    correction, NSTEPS)
                mc_mean_mwl = np.mean(self.lnMwl[total_mask])

                print(f"Theory:{theory_mwl_given_lam_sz} MC:{mc_mean_mwl}")

                diff = (theory_mwl_given_lam_sz - mc_mean_mwl)
                diff_array[i][j] = diff

        return (lam_array, SZ_array, diff_array, count_array)

        #plot x axis r, bin by SZ and lambda
        #plot x axis lambda, bin by SZ
        #y axis difference