import numpy as np

from fws import exphet_from_maf, get_bin_indices, maf_from_AD, maf_from_GT

def test_maf_from_AD():
    # four samples, N sites, two alleles:
    AD = np.array([[[10,0], [10,0], [0,10], [0,10]],
                    [[10,0], [np.nan,np.nan], [0,10], [np.nan,np.nan]], # -1 and np.nan are both classed as missing data
                    [[10,0], [-1,-1], [0,10], [-1,-1]],
                    [[-1,-1], [np.nan,np.nan], [-1,-1], [np.nan,np.nan]], # all missing, should return np.nan
                   ]) 
    
    mafs = maf_from_AD(AD)

    desired_result = np.array([0.5,
                               0.5,
                               0.5,
                               np.nan])
    
    for i in range(AD.shape[0]):
        assert mafs[i] == desired_result[i] or (np.isnan(mafs[i]) and np.isnan(desired_result[i])) # np.nan == np.nan evaluates to False, which is why we need the extra logic here

def test_maf_from_GT():
    diploid_GT = np.array([[[1,1], [1,1], [0,0], [0,0]],
                           [[1,1], [np.nan,np.nan], [0,0], [np.nan,np.nan]], # -1 and np.nan are both classed as missing data
                           [[1,1], [-1,-1], [0,0], [-1,-1]],
                           [[-1,-1], [np.nan,np.nan], [-1,-1], [np.nan,np.nan]], # all missing, should return np.nan
                           [[2,2], [2,2], [2,2], [0,0]],
                           [[2,0], [2,0], [2,0], [2,0]],
                          ])

    mafs = maf_from_GT(diploid_GT)

    desired_result = np.array([0.5,
                               0.5,
                               0.5,
                               np.nan,
                               0.25,
                               0.5])
            
    for i in range(diploid_GT.shape[0]):
        assert mafs[i] == desired_result[i] or (np.isnan(mafs[i]) and np.isnan(desired_result[i])) # np.nan == np.nan evaluates to False, so need some extra logic here


def test_exphet_from_maf():
    mafs = np.array([0.5, 0.25, 0.3])
    eh = exphet_from_maf(mafs)
    assert (eh == np.array([0.5, 0.375, 0.42])).all()

def test_bin_indices():
    mafs = np.array([0.5, 0.25, 0.3, 0.1, 0.001, 0.01, 0.025, 0.35, np.nan, np.nan]) # np.nan shouldn't appear in the output
    bin_indices = get_bin_indices(mafs)
    
    assert len(bin_indices) == 10 # because there are ten bins between 0 and 0.5 in increments of 0.05

    desired_result = [np.array([4, 5, 6]),
                      np.array([3]),
                      np.array([]),
                      np.array([]),
                      np.array([1]),
                      np.array([2]),
                      np.array([]),
                      np.array([7]),
                      np.array([]),
                      np.array([0])]

    for i in range(10):
        assert (bin_indices[i] ==  desired_result[i]).all()
