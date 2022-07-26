import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm, rv_histogram
from scipy.interpolate import interp1d, interp2d
from colossus.cosmology import cosmology
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import copy
import scipy as sp


class MonteCarloObservables(object):
    """Monte Carlo observables given mass function, scaling relations, and correlation coefficient

    Args:
        object (_type_): _description_
    """
    def __init__(self,
                 mass_function=None,
                 scaling_relation=None,
                 r=None,
                 multiplier=None):

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

        plt.hist(np.exp(self.lnlam), bins=20)
        plt.title("Histogram of lam")
        plt.show()

        plt.hist(np.exp(self.lnMwl), bins=20)
        plt.title("Histogram of Mwl")
        plt.show()

        plt.hist(np.exp(self.lnSZ), bins=20)
        plt.title("Histogram of SZ")
        plt.show()

        self.beta = mf.beta

        rv = mv.rvs(size=self.nh * multiplier)
        x = rv[:, 0]
        y = rv[:, 1]

        gauss = norm(0, 1)
        z = gauss.rvs(size=self.nh * multiplier)

        self.lnlam_for_pdf = np.repeat(self.lnlam_mean,
                                       multiplier) + self.scatter_lam * x
        self.lnMwl_for_pdf = np.repeat(self.lnMwl_mean,
                                       multiplier) + self.scatter_Mwl * y
        self.lnSZ_for_pdf = np.repeat(self.lnSZ_mean,
                                      multiplier) + self.scatter_SZ * z

    def get_pdf_in_bin_by_interpolation(self, data, left_edge, right_edge):
        """Get pdf in observable bin by interpolating

        Args:
            data (_type_): _description_
            left_edge (_type_): _description_
            right_edge (_type_): _description_

        Returns:
            _type_: _description_
        """
        data_in_bin = np.ma.masked_outside(data, left_edge,
                                           right_edge).compressed()
        counts, bin_edges = np.histogram(data_in_bin, density=True, bins=20)
        bin_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        pdf = interp1d(bin_mid, counts, bounds_error=False, fill_value=0)
        plt.hist(data_in_bin, bins=20)
        plt.title("Histogram for Interpolation")
        plt.show()
        plt.plot(np.linspace(left_edge, right_edge, 1000),
                 pdf(np.linspace(left_edge, right_edge, 1000)))
        plt.title("PDF from interpolation")
        plt.show()
        return (pdf)

    def get_pdf_in_bin_by_rv_histogram(self, data, left_edge, right_edge):
        """Get pdf for rv.histogram
        Args:
            data (_type_): _description_
            left_edge (_type_): _description_
            right_edge (_type_): _description_

        Returns:
            _type_: _description_
        """

        data_in_bin = np.ma.masked_outside(data, left_edge,
                                           right_edge).compressed()
        rv = sp.stats.rv_histogram(np.histogram(data_in_bin, bins=50))
        plt.hist(data_in_bin)
        plt.title("Histogram of Observable")
        plt.show()
        plt.plot(
            np.linspace(data_in_bin.min(), data_in_bin.max(), 10000),
            rv.pdf(np.linspace(data_in_bin.min(), data_in_bin.max(), 10000)))
        plt.title("PDF Generated from the Histogram")
        plt.show()
        # return (rv.pdf)

        return (rv.pdf)

    def theory_calculate_mean_mwl_given_lam(self, lam, correction=True):
        """Calculate the mean lensing mass given lambda

        Args:
            lam (_type_): _description_
        """

        mu_lam = (lam - self.sr.pi_lam) / self.sr.alpha_lam
        sig_lam = self.sr.scatter_lam / self.sr.alpha_lam

        mu_guess = mu_lam

        beta = self.beta(mu_guess)

        mean_mu_lam = mu_lam - beta * sig_lam**2

        # if mu_guess <= 32:
        #     beta = self.beta(mu_lam)
        # else:
        #     beta = 1.7

        mean_mwl = self.sr.pi_Mwl + self.sr.alpha_Mwl * mean_mu_lam + self.r * self.sr.scatter_Mwl * (
            mu_lam - mean_mu_lam) / sig_lam

        mean_mwl = self.sr.pi_Mwl + self.sr.alpha_Mwl * mean_mu_lam + beta * self.r * self.scatter_Mwl * sig_lam

        return mean_mwl

    def theory_calculate_mean_mwl_given_lam_sz(self,
                                               lnlam,
                                               lnsz,
                                               correction=True):
        """Calculate the mean lensing mass given lambda and SZ

        Args:
            lam (_type_): _description_
            SZ (_type_): _description_
            correction (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        mu_lam = (lnlam - self.sr.pi_lam) / self.sr.alpha_lam
        mu_SZ = (lnsz - self.sr.pi_SZ) / self.sr.alpha_SZ

        sig_lam = self.sr.scatter_lam / self.sr.alpha_lam
        sig_SZ = self.sr.scatter_SZ / self.sr.alpha_SZ

        mu_guess = (mu_lam / (sig_lam)**2 + mu_SZ /
                    (sig_SZ)**2) / (1 / sig_lam**2 + 1 / sig_SZ**2)

        beta = self.beta(mu_guess)

        second_term = (self.sr.alpha_Mwl *
                       (mu_lam * sig_SZ**2 + mu_SZ * sig_lam**2 - beta *
                        sig_lam**2 * sig_SZ**2)) / (sig_SZ**2 + sig_lam**2)

        third_term = (self.r * self.sr.scatter_Mwl * sig_lam *
                      (mu_lam - mu_SZ + beta * sig_SZ**2)) / (sig_SZ**2 +
                                                              sig_lam**2)

        theory_mean_mwl_given_lam_sz = self.sr.pi_Mwl + second_term + third_term

        return (theory_mean_mwl_given_lam_sz)

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
        # print(lnlam1, lnlam2)
        lam_range, lam_step = np.linspace(lnlam1, lnlam2, NSTEPS, retstep=True)
        lam_mid = 0.5 * (lam_range[1:] + lam_range[:-1])
        # lam_p = self.lam_pdf(lam_mid)

        # norm_factor = np.sum(lam_p) * lam_step
        # print("The normalization factor is:", norm_factor)

        # plt.plot(lam_mid, lam_p)
        # plt.title("lam PDF to be put in the integral")
        # plt.show()

        integral = 0

        x_array = lam_mid
        y_array = [None] * len(lam_mid)
        norm_array = [None] * len(lam_mid)

        for i, lam in tqdm(enumerate(lam_mid)):
            p_lam = self.lam_pdf(lam)
            mean_mwl = self.theory_calculate_mean_mwl_given_lam(
                lam, correction=correction)
            y_array[i] = p_lam * mean_mwl
            norm_array[i] = p_lam

        integral = np.trapz(y_array, x_array) / np.trapz(norm_array, x_array)

        # print("Integral before renormalization", integral)
        # print("Integral after renormalization", integral)

        return integral

    # def mean_mwl_in_lam_sz_bin(self, lnlam1, lnlam2, lnsz1, lnsz2, correction,
    #                            NSTEPS, pdf):
    #     """Calculate the precise lensing mass given lambda and SZ bin

    #     Args:
    #         lam1 (_type_): _description_
    #         lam2 (_type_): _description_
    #         SZ1 (_type_): _description_
    #         SZ2 (_type_): _description_
    #         correction (_type_): _description_

    #     Returns:
    #         _type_: _description_
    #     """

    #     # norm_factor_lam = self.P_lam.cdf(lam2) - self.P_lam.cdf(lam1)
    #     # norm_factor_SZ = self.P_SZ.cdf(SZ2) - self.P_SZ.cdf(SZ1)

    #     # norm_factor_lam = self.lam_kde.integrate_box_1d(lam1, lam2)
    #     # norm_factor_sz = self.sz_kde.integrate_box_1d(sz1, sz2)

    #     lam_range, lam_step = np.linspace(lnlam1, lnlam2, NSTEPS, retstep=True)
    #     sz_range, sz_step = np.linspace(lnsz1, lnsz2, NSTEPS, retstep=True)

    #     lam_mid = 0.5 * (lam_range[1:] + lam_range[:-1])
    #     sz_mid = 0.5 * (sz_range[1:] + sz_range[:-1])

    #     integral_mesh = np.empty([len(lam_mid), len(sz_mid)])
    #     norm_mesh = np.empty([len(lam_mid), len(sz_mid)])
    #     lam_mesh = np.empty([len(lam_mid), len(sz_mid)])
    #     sz_mesh = np.empty([len(lam_mid), len(sz_mid)])

    #     for i, lnlam in tqdm(enumerate(lam_mid)):
    #         for j, lnsz in enumerate(sz_mid):

    #             p_lam = self.lam_pdf(lnlam)
    #             p_sz = self.sz_pdf(lnsz)

    #             lam_mesh[i][j] = lnlam
    #             sz_mesh[i][j] = lnsz

    #             integral_mesh[i][
    #                 j] = p_lam * p_sz * self.theory_calculate_mean_mwl_given_lam_sz(
    #                     lnlam, lnsz, correction=True)

    #             norm_mesh[i][j] = p_lam * p_sz

    #     integral = np.trapz(np.trapz(integral_mesh, lam_mid, axis=0),
    #                         sz_mid,
    #                         axis=0)
    #     norm = np.trapz(np.trapz(norm_mesh, lam_mid, axis=0), sz_mid, axis=0)

    #     # mesh = self.theory_calculate_mean_mwl_given_lam_sz(
    #     #     lam_range.reshape(-1, 1),
    #     #     SZ_range.reshape(1, -1),
    #     #     correction=correction) * self.P_lam(
    #     #         lam_range.reshape(-1, 1)) / norm_factor_lam * self.P_SZ(
    #     #             SZ_range.reshape(1, -1)) / norm_factor_SZ

    #     # mesh = self.theory_calculate_mean_mwl_given_lam_sz(
    #     #     lam_range.reshape(-1, 1),
    #     #     SZ_range.reshape(1, -1),
    #     #     correction=correction) * self.P_lam.pdf(lam_range.reshape(
    #     #         -1, 1)) * self.P_SZ.pdf(SZ_range.reshape(1, -1))

    #     # mesh_norm = self.P_lam.pdf(lam_range.reshape(-1, 1)) * self.P_SZ.pdf(
    #     #     SZ_range.reshape(1, -1))

    #     # integral = romb([romb(SZ, SZ_step) for SZ in mesh], lam_step)
    #     # integral = simps([simps(SZ, SZ_range) for SZ in mesh], lam_range)

    #     print(f"{integral=}")
    #     print(f"{norm=}")

    #     return integral / norm

    def mean_mwl_in_lam_sz_bin(self, lnlam1, lnlam2, lnsz1, lnsz2, correction,
                               NSTEPS, pdf):
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

        lam_mask = (self.lnlam > lnlam1) & (self.lnlam < lnlam2)
        sz_mask = (self.lnSZ > lnsz1) & (self.lnSZ < lnsz2)

        total_mask = lam_mask & sz_mask

        lnlam_mean = np.mean(self.lnlam[total_mask])
        lnsz_mean = np.mean(self.lnSZ[total_mask])

        theory_mwl = self.theory_calculate_mean_mwl_given_lam_sz(
            lnlam_mean, lnsz_mean)

        return theory_mwl

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

        self.lam_pdf = self.get_pdf_in_bin(self.lnlam_for_pdf, lnlam1, lnlam2)

        lam_mask = (self.lnlam > lnlam1) & (self.lnlam < lnlam2)
        count = np.sum(lam_mask)

        theory_mwl_given_lam_bin = self.mean_mwl_in_lam_bin(
            lnlam1, lnlam2, correction, NSTEPS)
        mc_mean_mwl_given_lam_bin = np.mean(self.lnMwl[lam_mask])

        # print(
        # f"Theory:{theory_mwl_given_lam_bin} MC:{mc_mean_mwl_given_lam_bin}"
        # )

        diff = theory_mwl_given_lam_bin - mc_mean_mwl_given_lam_bin

        return (diff, count)

    def get_get_lam_sz_pdf(self, lnlam1, lnlam2, lnsz1, lnsz2):
        lnlam_mask = (self.lnlam > lnlam1) & (self.lnlam < lnlam2)
        lnsz_mask = (self.lnSZ > lnsz1) & (self.lnSZ < lnsz2)

    def mc_calculate_mean_mwl_diff_given_lam_sz_bin(self, correction, lam1,
                                                    lam2, sz1, sz2, NSTEPS,
                                                    pdf):
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

        # lnlam_mask = (self.lnlam_for_pdf > lnlam1) & (self.lnlam_for_pdf <
        #                                               lnlam2)
        # lnsz_mask = (self.lnSZ_for_pdf > lnsz1) & (self.lnSZ_for_pdf < lnsz2)

        # total_mask = lnlam_mask & lnsz_mask

        # def zero_pdf(x):
        #     return (0)

        # if np.sum(total_mask) != 0:
        #     if pdf == "rv_histogram":
        #         self.lam_pdf = self.get_pdf_in_bin_by_rv_histogram(
        #             self.lnlam_for_pdf[total_mask], lnlam1, lnlam2)
        #         self.sz_pdf = self.get_pdf_in_bin_by_rv_histogram(
        #             self.lnSZ_for_pdf[total_mask], lnsz1, lnsz2)

        #     elif pdf == "histogram_interpolation":
        #         self.lam_pdf = self.get_pdf_in_bin_by_interpolation(
        #             self.lnlam_for_pdf[total_mask], lnlam1, lnlam2)
        #         self.sz_pdf = self.get_pdf_in_bin_by_interpolation(
        #             self.lnSZ_for_pdf[total_mask], lnsz1, lnsz2)
        #     else:
        #         raise TypeError
        # else:
        #     self.lam_pdf = zero_pdf
        #     self.sz_pdf = zero_pdf

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

        # print("Lam bounds are", np.exp(lnlam1), np.exp(lnlam2))
        # print("SZ bounds are", np.exp(lnsz1), np.exp(lnsz2))

        theory_mwl_given_lam_sz = self.mean_mwl_in_lam_sz_bin(
            lnlam1,
            lnlam2,
            lnsz1,
            lnsz2,
            correction=correction,
            NSTEPS=NSTEPS,
            pdf=pdf)
        mc_mean_mwl = np.mean(self.lnMwl[total_mask])

        # print(f"Theory:{theory_mwl_given_lam_sz} MC:{mc_mean_mwl}")

        diff = (theory_mwl_given_lam_sz - mc_mean_mwl)

        # print("The count in the bin is", count)
        # print("The log diff is", diff)
        # print("The percentage error is", (np.exp(diff) - 1) * 100, "%")

        return (diff, count)

    def verify_theory_mean_mwl_given_lam_sz_bin(self, lam1, lam2, sz_threshold,
                                                bin_numbers, NSTEPS, pdf):
        """Verify that the theoretical formula is correct with different bin numbers

        Args:
            lam1 (_type_): _description_
            lam2 (_type_): _description_
            sz1 (_type_): _description_
            sz2 (_type_): _description_
            bin_numbers (_type_): _description_
            NSTEPS (_type_): _description_
        """

        args = {
            "lam1": lam1,
            "lam2": lam2,
            "sz_threshold": sz_threshold,
            "bin_numbers": bin_numbers,
            "NSTEPS": NSTEPS,
            "pdf": pdf
        }

        lam_list = [None] * len(bin_numbers)
        sz_list = [None] * len(bin_numbers)
        diff_list = [None] * len(bin_numbers)
        count_list = [None] * len(bin_numbers)

        for i, bin_number in enumerate(bin_numbers):

            diff_array = np.empty([bin_number, 2])
            count_array = np.empty([bin_number, 2])

            lam_range = np.linspace(lam1, lam2, bin_number + 1)
            sz_range = np.array([0.001, sz_threshold, 100])

            lam_mid = 0.5 * (lam_range[1:] + lam_range[:-1])
            sz_mid = 0.5 * (sz_range[1:] + sz_range[:-1])

            for j in range(len(lam_mid)):
                for k in range(len(sz_mid)):
                    diff_array[j][k], count_array[j][
                        k] = self.mc_calculate_mean_mwl_diff_given_lam_sz_bin(
                            correction=True,
                            lam1=lam_range[j],
                            lam2=lam_range[j + 1],
                            sz1=sz_range[k],
                            sz2=sz_range[k + 1],
                            NSTEPS=NSTEPS,
                            pdf=pdf)

            lam_list[i], sz_list[i], diff_list[i], count_list[
                i] = lam_mid, sz_mid, diff_array, count_array

        return (args, lam_list, sz_list, diff_list, count_list)

        # plot_diff(sz_list, r"SZ", sz1, sz2)

        #plot x axis r, bin by SZ and lambda
        #plot x axis lambda, bin by SZ
        #y axis difference)

    def plot_diff_by_bin_numbers(self, lam_list, sz_list, diff_list,
                                 count_list, **kwargs):
        """Plot the difference plot with different bin numbers

        Args:
            lam_list (_type_): _description_
            sz_list (_type_): _description_
            diff_list (_type_): _description_
            count_list (_type_): _description_
        """

        bin_numbers = kwargs['bin_numbers']
        lam1 = kwargs['lam1']
        lam2 = kwargs['lam2']

        def plot_diff(target_list, x_label, xlim1, xlim2):
            plt.figure(figsize=(10, 8), dpi=100)
            for i, target in enumerate(target_list):
                plt.plot(target_list[i],
                         diff_list[i][:, 0],
                         'x-',
                         label=f"{bin_numbers[i]} bins. Non-detection")
                plt.plot(target_list[i],
                         diff_list[i][:, 1],
                         'o--',
                         label=f"{bin_numbers[i]} bins. Detection")
                plt.xlim(xlim1, xlim2)
                # plt.ylim(-0.05, 0.05)
                plt.title("Integration Formula")
                plt.xlabel(x_label)
                plt.ylabel(r"$lnM_{wl}$ Theory - Numerical")

                plt.axhline(0, c='gray', ls='--')
                plt.axhline(0.01, c='k', ls='--')
                plt.axhline(-0.01, c='k', ls='--')

                plt.legend()
            plt.show()

        for i, bin_number in enumerate(bin_numbers):
            print(f"Statistics for {bin_number} bins")
            print(f"{count_list[i]=}")
            print(f"{diff_list[i]=}")
            print("---------------------------------------")

        plot_diff(lam_list, r"$\lambda$", lam1, lam2)

        diff_1d_non_detection, diff_1d_detection = np.array([
            diff for diff_array in diff_list for diff in diff_array[:, 0]
        ]), np.array(
            [diff for diff_array in diff_list for diff in diff_array[:, 1]])

        count_1d_non_detection = np.array([
            count for count_array in count_list for count in count_array[:, 0]
        ])

        count_1d_detection = np.array([
            count for count_array in count_list for count in count_array[:, 1]
        ])

        plt.figure(figsize=(10, 8), dpi=100)
        plt.scatter(count_1d_non_detection,
                    diff_1d_non_detection,
                    marker="X",
                    color='r',
                    label="SPT Non-Detection")
        plt.scatter(count_1d_detection,
                    diff_1d_detection,
                    marker="o",
                    color='g',
                    label="SPT Detection")
        plt.plot(np.logspace(0, 6, 100), 0.6 / np.sqrt(np.logspace(0, 6, 100)))
        plt.plot(np.logspace(0, 6, 100),
                 -0.5 / np.sqrt(np.logspace(0, 6, 100)))
        plt.xscale('log')
        plt.axhline(0, ls='-')
        plt.axhline(-0.01, ls='--')
        plt.axhline(0.01, ls='--')
        plt.xlabel("Count in the Bin")
        plt.ylabel(r"$lnM_{wl}$ Theory - Numerical")
        plt.title("Error vs count")
        plt.legend()
        plt.show()
