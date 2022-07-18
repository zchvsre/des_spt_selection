import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_histogram
from scipy.interpolate import interp1d
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import copy


class MonteCarloObservables(object):
    """Monte Carlo observables given mass function, scaling relations, and correlation coefficient

    Args:
        object (_type_): _description_
    """
    def __init__(self, mass_function=None, scaling_relation=None, r=None):

        self.sr = scaling_relation
        mf = mass_function

        self.lnM = np.log(mf.mass[mf.mass > 1E13])
        self.nh = len(self.lnM)

        self.lnMwl_mean = copy.deepcopy(self.lnM)
        self.lnlam_mean = self.sr.alpha_lam * self.lnM + self.sr.pi_lam
        self.lnSZ_mean = self.sr.alpha_SZ * self.lnM + self.sr.pi_SZ

        self.r = r
        self.scatter_Mwl = self.sr.scatter_Mwl
        self.scatter_lam = self.sr.scatter_lam
        self.scatter_SZ = self.sr.scatter_SZ

        mv = multivariate_normal([0, 0], [[1, r], [r, 1]])
        rv = mv.rvs(size=self.nh)
        x = rv[:, 0]
        y = rv[:, 1]

        gauss = norm(0, 1)
        z = gauss.rvs(size=self.nh)

        self.lnlam = self.lnlam_mean + self.scatter_lam * x
        self.lnMwl = self.lnMwl_mean + self.scatter_Mwl * y
        self.lnSZ = self.lnSZ_mean + self.scatter_SZ * z

        plt.hist(self.lnlam, bins=20)
        plt.title("Histogram of lnlam")
        plt.show()

        plt.hist(self.lnMwl, bins=20)
        plt.title("Histogram of lnMwl")
        plt.show()

        plt.hist(self.lnSZ, bins=20)
        plt.title("Histogram of lnMwl")
        plt.show()

        self.beta = mf.beta


#         multiplier = 1000
#         rv = mv.rvs(size=nh * multiplier)
#         x = rv[:, 0]
#         y = rv[:, 1]

#         gauss = norm(0, 1)
#         z = gauss.rvs(size=nh * multiplier)

#         self.lnlam_for_pdf = np.repeat(lnlam_mean,
#                                        multiplier) + scatter_lam * x
#         self.lnMwl_for_pdf = np.repeat(lnMwl_mean,
#                                        multiplier) + scatter_Mwl * y
#         self.lnSZ_for_pdf = np.repeat(lnSZ_mean, multiplier) + scatter_SZ * z

# self.P_lam = rv_histogram(
#     np.histogram(self.lnlam,
#                  bins=np.linspace(np.log(1), np.log(50), 30)))
# self.P_SZ = rv_histogram(
#     np.histogram(self.lnSZ,
#                  bins=np.linspace(np.log(0.001), np.log(5), 100)))

# self.P_lam = rv_histogram(np.histogram(self.lnlam, bins=100))
# self.P_SZ = rv_histogram(np.histogram(self.lnSZ, bins=100))

    def theory_calculate_mean_mwl_given_lam(self, lam, correction=True):
        """Calculate the mean lensing mass given lambda

        Args:
            lam (_type_): _description_
        """

        mu_lam = (lam - self.sr.pi_lam) / self.sr.alpha_lam
        sig_lam = self.sr.scatter_lam / self.sr.alpha_lam

        mu_guess = mu_lam

        if mu_guess <= 32:
            beta = self.beta(mu_lam)
        else:
            beta = 1.7

        mean_mwl = self.sr.pi_Mwl + self.sr.alpha_Mwl * (
            mu_lam - beta * sig_lam**2) + self.r * self.sr.scatter_Mwl * (
                mu_lam - mu_lam - beta * sig_lam**2) / sig_lam

        return mean_mwl

    def theory_calculate_mean_mwl_given_lam_sz(self, lam, SZ, correction=True):
        """Calculate the mean lensing mass given lambda and SZ

        Args:
            lam (_type_): _description_
            SZ (_type_): _description_
            correction (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        mu_lam = (lam - self.sr.pi_lam) / self.sr.alpha_lam
        mu_SZ = (SZ - self.sr.pi_SZ) / self.sr.alpha_SZ

        sig_lam = self.sr.scatter_lam / self.sr.alpha_lam
        sig_SZ = self.sr.scatter_SZ / self.sr.alpha_SZ

        mu_guess = (mu_lam / (sig_lam)**2 + mu_SZ /
                    (sig_SZ)**2) / (1 / sig_lam**2 + 1 / sig_SZ**2)
        if mu_guess <= 32:
            beta = self.beta(mu_guess)
        else:
            beta = 1.7
        # print("beta", beta)

        mu_given_lam_SZ_num = (self.sr.alpha_lam / self.sr.scatter_lam**2) * (
            lam - self.sr.pi_lam) + (self.sr.alpha_SZ / self.sr.scatter_SZ**
                                     2) * (SZ - self.sr.pi_SZ) - beta
        mu_given_lam_SZ_den = (self.sr.alpha_lam / self.sr.scatter_lam)**2 + (
            self.sr.alpha_SZ / self.sr.scatter_SZ)**2
        mu_given_lam_SZ = mu_given_lam_SZ_num / mu_given_lam_SZ_den

        mu_given_lam = (lam - self.sr.pi_lam) / self.sr.alpha_lam
        mu_given_SZ = (SZ - self.sr.pi_SZ) / self.sr.alpha_SZ

        if correction is True:

            third_term_num = self.r * self.sr.scatter_Mwl * (
                self.sr.scatter_lam / self.sr.alpha_lam) * (
                    mu_given_lam - mu_given_SZ + beta *
                    (self.sr.scatter_SZ / self.sr.alpha_SZ)**2)
            third_term_den = (self.sr.scatter_SZ / self.sr.alpha_SZ)**2 + (
                self.sr.scatter_lam / self.sr.alpha_lam)**2
            third_term = third_term_num / third_term_den
        else:
            third_term = 0

        theory_mwl_given_lambda_SZ = self.sr.pi_Mwl + self.sr.alpha_Mwl * mu_given_lam_SZ + third_term

        return (theory_mwl_given_lambda_SZ)

    def mean_mwl_in_lam_bin(self, lnlam1, lnlam2, correction, NSTEPS):
        """Mean lensing mass in a richness bin 

        Args:
            lnlam1 (_type_): _description_
            lnlam2 (_type_): _description_
            correction (_type_): _description_
            NSTEPS (_type_): _description_

        Returns:
            _type_: _description_
        """
        lam_range, lam_step = np.linspace(lnlam1, lnlam2, NSTEPS, retstep=True)
        lam_p = self.lam_pdf(lam_range)

        norm_factor = np.sum(lam_p) * lam_step
        print("The normalization factor is:", norm_factor)

        plt.plot(lam_range, lam_p)
        plt.title("lam PDF to be put in the integral")
        plt.show()

        integral = 0

        for i, lam in tqdm(enumerate(lam_range)):
            p_lam = lam_step * lam_p[i]
            mean_mwl = self.theory_calculate_mean_mwl_given_lam(
                lam, correction=correction)
            integral += p_lam * mean_mwl

        print("Integral before renormalization", integral)
        integral /= norm_factor
        print("Integral after renormalization", integral)

        return integral

    def mean_mwl_in_lam_sz_bin(self, lnlam1, lnlam2, lnsz1, lnsz2, correction,
                               NSTEPS):
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

        lam_range, lam_step = np.linspace(lnlam1, lnlam2, NSTEPS, retstep=True)
        sz_range, sz_step = np.linspace(lnsz1, lnsz2, NSTEPS, retstep=True)

        # lam_p = self.lam_kde.pdf(lam_range)
        # sz_p = self.sz_kde.pdf(sz_range)

        lam_p = self.lam_pdf(lam_range)
        sz_p = self.sz_pdf(sz_range)

        # print("The number of out of bound points are", np.sum(lam_p == 0),
        #       np.sum(sz_p == 0))
        # lam_p[0] = (lam_p[1] - lam_p[2]) + lam_p[1]
        # lam_p[-1] = (lam_p[-2] - lam_p[-3]) + lam_p[-2]
        # sz_p[0] = (sz_p[1] - sz_p[2]) + sz_p[1]
        # sz_p[-1] = (sz_p[-2] - sz_p[-3]) + sz_p[-2]
        # print("The number of out of bound points are", np.sum(lam_p == 0),
        #       np.sum(sz_p == 0))

        norm_factor_lam = np.sum(lam_p) * lam_step
        norm_factor_sz = np.sum(sz_p) * sz_step

        print("The normalization factors are:", norm_factor_lam,
              norm_factor_sz)

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

        print("Integral before renormalization", integral)
        integral /= norm_factor_lam
        integral /= norm_factor_sz
        print("Integral after renormalization", integral)

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

    def get_pdf_in_bin(self, data, left_edge, right_edge):
        """Get pdf in observable bin

        Args:
            data (_type_): _description_
            left_edge (_type_): _description_
            right_edge (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_in_bin = np.ma.masked_outside(data, left_edge,
                                           right_edge).compressed()
        counts, bin_edges = np.histogram(data_in_bin, density=True, bins=10)
        bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        pdf = interp1d(bin_mid,
                       counts,
                       bounds_error=False,
                       fill_value="extrapolate")
        plt.hist(data_in_bin, bins=20)
        plt.title("Histogram for Interpolation")
        plt.show()
        plt.plot(np.linspace(left_edge, right_edge, 1000),
                 pdf(np.linspace(left_edge, right_edge, 1000)))
        plt.title("PDF from interpolation")
        plt.show()
        return (pdf)

    def mc_calculate_mean_mwl_diff_given_lam_bin(self,
                                                 lam1,
                                                 lam2,
                                                 NSTEPS,
                                                 correction=True):
        """Calculate the mean lensing mass difference given lambda

        Args:
            nbins (_type_): _description_
            correction (_type_): _description_
            lam1 (_type_): _description_
            lam2 (_type_): _description_
            NSTEPS (_type_): _description_
        """

        lnlam1, lnlam2 = np.log(lam1), np.log(lam2)

        self.lam_pdf = self.get_pdf_in_bin(self.lnlam, lnlam1, lnlam2)

        lam_mask = (self.lnlam > lnlam1) & (self.lnlam < lnlam2)
        count = np.sum(lam_mask)

        theory_mwl_given_lam_bin = self.mean_mwl_in_lam_bin(
            lnlam1, lnlam2, correction, NSTEPS)
        mc_mean_mwl_given_lam_bin = np.mean(self.lnMwl[lam_mask])

        print(
            f"Theory:{theory_mwl_given_lam_bin} MC:{mc_mean_mwl_given_lam_bin}"
        )

        diff = theory_mwl_given_lam_bin - mc_mean_mwl_given_lam_bin

        return (diff, count)

    def mc_calculate_mean_mwl_diff_given_lam_sz_bin(self, correction, lam1,
                                                    lam2, sz1, sz2, NSTEPS):
        """Calculate the mean lensing mass difference given lambda and SZ in Monte Carlo.

        Args:
            nbins (_type_): _description_
            correction (_type_): _description_

        Returns:
            _type_: _description_
        """

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

        # pdf from rv_histogram
        # def get_pdf_in_bin(data, left_edge, right_edge):

        #     data_in_bin = np.ma.masked_outside(data, left_edge,
        #                                        right_edge).compressed()
        #     rv = sp.stats.rv_histogram(np.histogram(data_in_bin, bins=500))
        #     plt.hist(data_in_bin)
        #     plt.title("Histogram of Observable")
        #     plt.show()
        #     plt.plot(
        #         np.linspace(data_in_bin.min(), data_in_bin.max(), 10000),
        #         rv.pdf(np.linspace(data_in_bin.min(), data_in_bin.max(),
        #                            10000)))
        #     plt.title("PDF Generated from the Histogram")
        #     plt.show()
        #     # return (rv.pdf)
        #     return (rv.pdf)

        # pdf from interpolating histogram

        lnlam1, lnlam2 = np.log(lam1), np.log(lam2)
        lnsz1, lnsz2 = np.log(sz1), np.log(sz2)

        self.lam_pdf = self.get_pdf_in_bin(self.lnlam, lnlam1, lnlam2)
        self.sz_pdf = self.get_pdf_in_bin(self.lnSZ, lnsz1, lnsz2)

        # pdf from kde estimate
        # def get_pdf_in_bin(data, left_edge, right_edge):
        #     data_in_bin = np.ma.masked_outside(data, left_edge,
        #                                        right_edge).compressed()

        #     kde = sp.stats.gaussian_kde(data_in_bin)

        #     plt.hist(data_in_bin)
        #     plt.show()
        #     plt.plot(np.linspace(left_edge, right_edge, 1000),
        #              kde.pdf(np.linspace(left_edge, right_edge, 1000)))
        #     plt.show()
        #     return (kde)

        # lnlam_bins = pd.qcut(self.lnlam,nbins,retbins=True)[1]
        # lnSZ_bins = pd.qcut(self.lnSZ,nbins,retbins=True)[1]

        SZ_mask = (self.lnSZ > lnsz1) & (self.lnSZ <= lnsz2)
        lam_mask = (self.lnlam > lnlam1) & (self.lnlam <= lnlam2)

        total_mask = SZ_mask & lam_mask  #combine the richness and SZ mask
        count = np.sum(total_mask)

        print("Lam bounds are", np.exp(lnlam1), np.exp(lnlam2))
        print("SZ bounds are", np.exp(lnsz1), np.exp(lnsz2))

        theory_mwl_given_lam_sz = self.mean_mwl_in_lam_sz_bin(
            lnlam1, lnlam2, lnsz1, lnsz2, correction=correction, NSTEPS=NSTEPS)
        mc_mean_mwl = np.mean(self.lnMwl[total_mask])

        print(f"Theory:{theory_mwl_given_lam_sz} MC:{mc_mean_mwl}")

        diff = (theory_mwl_given_lam_sz - mc_mean_mwl)

        return (diff, count)

        #plot x axis r, bin by SZ and lambda
        #plot x axis lambda, bin by SZ
        #y axis difference