# BAOfit
Code fits to the BAO shape of catalog data

Install anaconda

Install emcee, jupyter, and nbodykit python packages

https://emcee.readthedocs.io/en/stable/

https://nbodykit.readthedocs.io/en/latest/

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
* The Window and Wide angle matrices must have the same k spacing as the data/covariance
* If using a "combined" fit (i.e. 2 separate data sets/covariances) the input covariance must be one block-diagonal covariance.
* As of v0.1, no monopole only fits

