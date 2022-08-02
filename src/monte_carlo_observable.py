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
    def __init__(self, mass_function=None, scaling_relation=None, r=None):

        self.sr = scaling_relation
        mf = mass_function

        self.lnM = np.log(mf.mass[mf.mass > 1E13])
        self.nh = len(self.lnM)

        print("Total number of massive halos:", self.nh)

        self.ln_mwl_mean = copy.deepcopy(self.lnM)
        self.ln_lam_mean = self.sr.alpha_lam * self.lnM + self.sr.pi_lam
        self.ln_sz_mean = self.sr.alpha_SZ * self.lnM + self.sr.pi_SZ

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

        self.ln_lam = self.ln_lam_mean + self.scatter_lam * x
        self.ln_mwl = self.ln_mwl_mean + self.scatter_Mwl * y
        self.ln_sz = self.ln_sz_mean + self.scatter_SZ * z

        plt.hist(np.exp(self.ln_lam), bins=np.linspace(0, 100, 20))
        plt.title("Histogram of lam")
        plt.xscale('log')
        plt.yscale('log')
        plt.show()

        plt.hist(np.exp(self.ln_mwl), bins=20)
        plt.title("Histogram of Mwl")
        plt.show()

        plt.hist(np.exp(self.ln_sz), bins=np.linspace(0, 100, 40))
        plt.title("Histogram of SZ")
        plt.yscale('log')
        plt.show()
        print("Number of halos with SZ < 4:", np.sum(self.ln_sz < np.log(4)))
        print("Number of halos with SZ > 4:", np.sum(self.ln_sz > np.log(4)))

        self.beta = mf.beta

    def theory_calculate_mean_mwl_given_lam_sz(self, lnlam, lnsz):
        """Calculate the mean lensing mass given lambda and SZ

        Args:
            lam (_type_): _description_
            SZ (_type_): _description_
            correction (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        mu_lam = (lnlam - self.sr.pi_lam) / self.sr.alpha_lam
        mu_sz = (lnsz - self.sr.pi_SZ) / self.sr.alpha_SZ

        sig_lam = self.sr.scatter_lam / self.sr.alpha_lam
        sig_sz = self.sr.scatter_SZ / self.sr.alpha_SZ

        mu_guess = (mu_lam / (sig_lam)**2 + mu_sz /
                    (sig_sz)**2) / (1 / sig_lam**2 + 1 / sig_sz**2)

        beta = self.beta(mu_guess)

        second_term = (self.sr.alpha_Mwl *
                       (mu_lam * sig_sz**2 + mu_sz * sig_lam**2 - beta *
                        sig_lam**2 * sig_sz**2)) / (sig_sz**2 + sig_lam**2)

        third_term = (self.r * self.sr.scatter_Mwl * sig_lam *
                      (mu_lam - mu_sz + beta * sig_sz**2)) / (sig_sz**2 +
                                                              sig_lam**2)

        theory_mean_mwl_given_lam_sz = self.sr.pi_Mwl + second_term + third_term

        return (theory_mean_mwl_given_lam_sz)

    def mean_mwl_in_lam_sz_bin(self, lnlam1, lnlam2, lnsz1, lnsz2):
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

        lam_mask = (self.ln_lam > lnlam1) & (self.ln_lam < lnlam2)
        sz_mask = (self.ln_sz > lnsz1) & (self.ln_sz < lnsz2)

        total_mask = lam_mask & sz_mask

        lnlam_mean = np.mean(self.ln_lam[total_mask])
        lnsz_mean = np.mean(self.ln_sz[total_mask])

        theory_mwl = self.theory_calculate_mean_mwl_given_lam_sz(
            lnlam_mean, lnsz_mean)

        return theory_mwl

    def mc_calculate_mean_mwl_diff_given_lam_sz_bin(self, lam1, lam2, sz1,
                                                    sz2):
        """Calculate the mean lensing mass difference given lambda and SZ in Monte Carlo.

        Args:
            nbins (_type_): _description_
            correction (_type_): _description_

        Returns:
            _type_: _description_
        """

        lnlam1, lnlam2 = np.log(lam1), np.log(lam2)
        lnsz1, lnsz2 = np.log(sz1), np.log(sz2)

        sz_mask = (self.ln_sz > lnsz1) & (self.ln_sz < lnsz2)
        lam_mask = (self.ln_lam > lnlam1) & (self.ln_lam < lnlam2)

        total_mask = sz_mask & lam_mask  #combine the richness and SZ mask
        count = np.sum(total_mask)

        mc_mean_mwl = np.mean(self.ln_mwl[total_mask])

        # print("Lam bounds are", np.exp(lnlam1), np.exp(lnlam2))
        # print("SZ bounds are", np.exp(lnsz1), np.exp(lnsz2))

        theory_mwl_given_lam_sz = self.mean_mwl_in_lam_sz_bin(
            lnlam1, lnlam2, lnsz1, lnsz2)

        # print(f"Theory:{theory_mwl_given_lam_sz} MC:{mc_mean_mwl}")

        diff = (theory_mwl_given_lam_sz - mc_mean_mwl)

        # print("The count in the bin is", count)
        # print("The log diff is", diff)
        # print("The percentage error is", (np.exp(diff) - 1) * 100, "%")

        return (diff, count)

    def verify_theory_mean_mwl_given_lam_sz_bin(self, lam1, lam2, sz_threshold,
                                                bin_numbers):
        """Verify that the theoretical formula is correct with different bin numbers

        Args:
            lam1 (_type_): _description_
            lam2 (_type_): _description_
            sz1 (_type_): _description_
            sz2 (_type_): _description_
            bin_numbers (_type_): _description_
            NSTEPS (_type_): _description_
        """

        kwargs = {
            "lam1": lam1,
            "lam2": lam2,
            "sz_threshold": sz_threshold,
            "bin_numbers": bin_numbers
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
                            lam1=lam_range[j],
                            lam2=lam_range[j + 1],
                            sz1=sz_range[k],
                            sz2=sz_range[k + 1])

            lam_list[i], sz_list[i], diff_list[i], count_list[
                i] = lam_mid, sz_mid, diff_array, count_array

        return (kwargs, lam_list, sz_list, diff_list, count_list)

    # def verify_narrow_bin(bin_sizes):

    #     diff_list = [None]*len(bin_sizes)

    #     max_lnm = np.max(self.lnM)
    #     min_lnm = np.min(self.lnM)

    #     for bin_size in bin_sizes:
    #         bin_grid = np.arange(min_lnm, max_lnm, bin_size)

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
                plt.plot(target,
                         diff_list[i][:, 0],
                         'x-',
                         label=f"{bin_numbers[i]} bins. SPT Non-Detection")
                plt.plot(target,
                         diff_list[i][:, 1],
                         'o--',
                         label=f"{bin_numbers[i]} bins. SPT Detection")
                plt.xlim(xlim1, xlim2)
                plt.title(r"Comparison of Analytic $M_{wl}$ in $\lambda$ Bins")
                plt.xlabel(x_label)
                plt.ylabel(r"$lnM_{wl}$ Theory - Numerical")

                plt.axhline(0, c='gray', ls='--')
                plt.axhline(0.01, c='k', ls='--')
                plt.axhline(-0.01, c='k', ls='--')

                plt.legend()
            plt.show()

        for i, bin_number in enumerate(bin_numbers):
            print(
                "All halos within lam1 and lam2:",
                np.sum((self.ln_lam > np.log(lam1))
                       & (self.ln_lam < np.log(lam2))))
            print(f"Statistics for {bin_number} bins")
            print(f"{count_list[i]=}")
            print(f"{diff_list[i]=}")
            print("---------------------------------------")

        plot_diff(lam_list, r"Mean $\lambda$ in Bin", lam1, lam2)

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
