
# Fws



## Installation

Supported Python versions are 3.10 and 3.11:

```
python3.11 -m pip install git+https://github.com/bjeight/fws
```

Fws will run much faster if you have numba installed:

```
python3.11 -m pip install numba
```

## Use

```
> Python
Python 3.11.6 (main, Nov 30 2023, 12:53:53) [Clang 15.0.0 (clang-1500.0.40.1)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> from fws import fws
>>> help(fws)
Help on function fws in module fws:

fws(a: typing.Annotated[numpy.ndarray[typing.Any, numpy.dtype[~DType]], typing.Literal['N', 2]], input: str = 'AD') -> <built-in function array>
    Calculate Fws for each sample in a population, from an array of biallelic allele depth or genotype information

    Parameters
    ----------
    a : ArrayNxNx2
        numpy.array with shape: (n_sites x n_samples x 2) containing allele depth (AD)
        or genotype (GT) information at biallelic sites for each sample.
    input: str = "AD" [| "GT"]
        specifies what the contents of `a` are - either the allele depth for each allele
        for each sample at each site (the default), or diploid genotype calls.

    Returns
    -------
    np.array[float]
        An array of fws values, in the same order as the samples in the input
```