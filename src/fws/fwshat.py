import numpy as np
import statsmodels as sm

from fwstypes import ArrayNxNx2
from fws import maf_from_AD, maf_from_GT, get_bin_indices, exphet_from_maf

#  1 - (individual / population)

def fwshat(
    a: ArrayNxNx2, # n_sites x n_samples x 2 array of allele depths (AD)
    input: str = "AD",
):
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
        p = [het for het,idx in zip(pop_exphet_binned, non_nan_index) if idx]
        s = [het for het,idx in zip(samp_exp_het_binned, non_nan_index) if idx]

        x = [pp - ss for pp, ss in zip(p, s)]
        y = [pp + ss for pp, ss in zip(p, s)]
    
        # then fit the linear regression:
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