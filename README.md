# BAOfit
Code fits to the BAO shape of catalog data

Install anaconda

Install emcee, jupyter, and nbodykit python packages


https://emcee.readthedocs.io/en/stable/user/install/

https://nbodykit.readthedocs.io/en/latest/getting-started/install.html

Set up a conda environment

1. conda create --n myenv python=3.7
2. conda activate myenv
3. conda install -c conda-forge emcee
4. conda install -c bccp nbodykit
5. conda install -c anaconda configobj

## Guide

Set up the fitting configuration in the config.ini file

This will include 
* Cosmology parameters
* Paths to input/output files
* Parameters for fitting (e.g. kmin,kmax,multipoles)

A few other considerations
* The minimum k specified in the config file must be greater than or equal to the minimum k of the covariance matrix
* Window function convolution is done through matrix multiplication of the
model, and as of v0.1 has only been tested with dk=0.01
* If using a "combined" fit (i.e. 2 separate data sets/covariances) the input
covariance must be one block-diagonal covariance.  If using a combined fit and
convolution, the same goes for the window and wide-angle matrices
* As of v0.1, no monopole only fits

