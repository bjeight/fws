from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels as sm

# Define a dummy decorator for the case where numba isn't available:
#   the decorated function will fall back to the pure python implementation.
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

from fwstypes import ArrayNxNx2


__all__ = ["fws"]

def fws(
    a: ArrayNxNx2, # n_sites x n_samples x 2 array of allele depths (AD)
    input: str = "AD",
) -> list["float"]:
    """
    Calculate Fws for each sample in a population, from an array of biallelic allele depth information

    Parameters
    ----------
    a : ArrayNxNx2
        numpy.array with shape: (n_sites x n_samples x 2) containing allele depth (AD) 
        or genotype (GT) information at biallelic sites for each sample
    input: str
        "AD" | "GT"

    Returns
    -------
    list[float]
        A list of fws values, in the same order as the samples in the input
   """
    
    if input == "AD":
        AD = True
    elif input == "GT":
        AD = False
    else:
        raise Exception("Unknown input type provided to fws")

    # Calculate population-level minor allele frequency (MAF)
    if AD:
        pop_maf = maf_from_AD(a)
    else:
        pop_maf = maf_from_GT(a)

    # Bin the population-level MAFs and calculate mean population-level expected heterozygosity per bin
    bin_indices = get_bin_indices(pop_maf)
    pop_mafs_binned = [pop_maf[b] for b in bin_indices]
    pop_exphet_binned = [np.nanmean(exphet_from_maf(x)) for x in pop_mafs_binned]

    # Then perform the sample-level calculation of Fws
    fws = np.zeros(a.shape[1])
    # For every sample
    for i in range(a.shape[1]):
        # calculate sample-level MAF
        if AD:
            samp_maf = maf_from_AD(a[:,[i],:])
        else:
            samp_maf = maf_from_GT(a[:,[i],:])

        # bin the sample-level MAFs using the population bin indices
        samp_maf_binned = [samp_maf[b] for b in bin_indices]
        
        # and calculate sample-level expected heterozygosity
        samp_exp_het_binned = [np.nanmean(exphet_from_maf(x)) for x in samp_maf_binned]

        # Get an index such that there are no nans in either pop or sample vector
        non_nan_index = ~np.isnan(pop_exphet_binned) & ~np.isnan(samp_exp_het_binned)
        x = [het for het,idx in zip(pop_exphet_binned, non_nan_index) if idx]
        y = [het for het,idx in zip(samp_exp_het_binned, non_nan_index) if idx]
        
        # then fit the linear regression: samp_exp_het_binned ~ pop_exp_het_binned.
        # The code below doesn't fit an intercept, so it is constrained to be 0
        model = sm.OLS(y, x)
        slope = model.fit().params[0]
        
        # Fiddle the fws value a little, depending on the regression
        if slope < 0:
            f = np.nan
        elif slope > 1:
            f = 0
        else:
            f = 1-slope
                
        # and add it to the list to be returned
        fws[i] = f

    return fws

@njit()
def maf_from_AD(
        AD: ArrayNxNx2, # n_sites x n_samples x 2 array of allele depths (AD)
) -> np.array: # 1d array of minor allele frequencies, len(n_sites)
    
    # We will populate this array with mafs
    mafs = np.zeros(AD.shape[0], dtype=np.float64)

    # For every site
    for i in range(AD.shape[0]):

        # Initiate a vector a per-sample AFs to calculate mean MAF from
        AFs = np.zeros(AD.shape[1], dtype=np.float64)

        # For every sample
        for j in range(AD.shape[1]):
            
            ad1 = AD[i,j,0]
            ad2 = AD[i,j,1]
            
            # skip sites with missing data
            if ad1 < 0 or ad2 < 0 or np.isnan(ad1) or np.isnan(ad2):
                AFs[j] = np.nan
                continue

            if (ad1 + ad2) == 0:
                AFs[j] = np.nan
                continue
            else:
                AFs[j] = ad1 / (ad1 + ad2)

        # Calculate maf from this site's average AFs
        mafs[i] = np.nanmean(AFs)
        if mafs[i] > 0.5:
            mafs[i] = 1 - mafs[i]

    return mafs

@njit()
def maf_from_GT(
        GT: ArrayNxNx2 # n_sites x n_samples x 2 array of genotypes (GT)
) -> np.array: # 1d array of minor allele frequencies, len(n_sites)
    
    # We will populate this array with mafs
    mafs = np.zeros(GT.shape[0], dtype=np.float64)

    # For every site
    for i in range(GT.shape[0]):
        # Initiate a dict of zeroed counts of each allele
        d = defaultdict(int)
        # For every sample
        for j in range(GT.shape[1]):
            # skip missing data
            if GT[i,j,0] < 0 or GT[i,j,1] < 0:
                continue
            # add a count of each allele present to the dictionary
            d[GT[i,j,0]] += 1
            d[GT[i,j,1]] += 1

        # a list of allele counts to populate with the dictionary information
        AC = []
        # append the dict values to the list
        for v in d.values():
            AC.append(v)
        # if there is only missing data at this site, set maf to nan and continue to the next site
        if len(AC) == 0:
            mafs[i] = np.nan
            continue
        # There might be only one allele if this function is called at the sample level.
        # To prevent an index out of bounds error, just append a 0 (allele count) to the list,
        # which will still result in the correct MAF (0.0)
        elif len(AC) == 1:
            AC.append(0)
        # If we sort the list, the frequency we calculate below will always be < 0.5 because
        # the lower allele count is l[0]. This means we don't have to check for MAF > 0.5
        AC.sort()

        # calculate maf as lowest allele count / total allele count
        maf = AC[0] / (AC[0] +  AC[1])

        mafs[i] = maf

    return mafs

# Calculate expected heterozygosity from minor allele frequency
def exphet_from_maf(
    mafs: np.ndarray, # a 1d array of floats which are minor allele frequencies
) -> np.ndarray: # the operation is vectorised so the lenth of the output is the same as len(mafs)
    # expected heterozygosity is the same as 1 - sum(expected homozygosity for each allele):
    # exp_het = 1 - (((1-mafs)**2) + mafs**2)
    # or, equivalently, for a biallelic site: 2 * p * q (like in the Hardy-Weinberg equation):
    exp_het = 2 * (1-mafs) * mafs
    return exp_het
   
# Given a 1d array of minor allele frequencies, return the indices which will place them in 10 equally-spaced bins in [0,0.5]
def get_bin_indices(
    mafs: np.ndarray, # a 1d array of mafs
) -> list[np.ndarray]: # a list of indices for binning the maf array
    bin_boundaries = np.linspace(0, 0.5, 11) # [0.05,0.10,0.15,0.20,..,0.45,0.50]
    bins = pd.cut(mafs, bins=bin_boundaries, include_lowest = True) # splits mafs into the bins whose boundaries are defined above
    bin_indices = [np.where(bins == b)[0] for b in bins.categories] # a (length = len(mafs)) list of indices which places mafs into the 10 bins defined above
    return bin_indices