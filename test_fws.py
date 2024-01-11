import numpy as np

from fws import exphet_from_maf, get_bin_indices

def test_exphet_from_maf():
    mafs = np.array([0.5, 0.25, 0.3])
    eh = exphet_from_maf(mafs)
    assert (eh == np.array([0.5, 0.375, 0.42])).all()

def test_bin_indices():
    mafs = np.array([0.5, 0.25, 0.3, 0.1, 0.001, 0.01, 0.025])
    bin_indices = get_bin_indices(mafs)
    
    assert len(bin_indices) == 10

    desired_result = [np.array([4, 5, 6]),
                      np.array([3]),
                      np.array([]),
                      np.array([]),
                      np.array([1]),
                      np.array([2]),
                      np.array([]),
                      np.array([]),
                      np.array([]),
                      np.array([0])]

    for i in range(10):
        assert (bin_indices[i] ==  desired_result[i]).all()
        
# test_bin_indices()