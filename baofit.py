
import emcee
import numpy as np
from nbodykit.lab import *
from numpy.linalg import inv
import scipy.integrate as integrate
import math
import scipy.optimize as op
from scipy.optimize import curve_fit
import numpy.linalg as linalg
from multiprocessing import Pool
import tqdm
import h5py
from configobj import ConfigObj
from scipy.interpolate import InterpolatedUnivariateSpline as IUS

import sys


def chi2f(params):

    modelP = model(params,combined)
    
    
    if combined:
            hs = size//2
    else:
            hs = size
                        
    Hart = (1000-hs-2)/(1000-1)

    covinvhart = covinv*Hart
    Pkmodel = modelP.run(km)

    vec = Pkdata - Pkmodel
    
    chisq = np.dot(vec,np.dot(covinvhart,vec))

    if not np.isfinite(chisq):
         return np.inf

    return chisq-log_prior(params)
    #return -0.5*chisq + log_prior(params)



def lnlh(params):
        chi2 = chi2f(params)
        return -0.5*chi2 + log_prior(params)


def log_prior(params):
    if combined:
            B = params[0]
            #beta = params[1]
            alpha_perp = params[1]
            alpha_par = params[2]
            #f1 = params[3]
            B2 = params[3]
            #f2 = params[5]

            f1 = 1.0
            f2 = 1.0
                                        
            if 0.8 < alpha_perp < 1.2 and 0.8 < alpha_par < 1.2 and 0.0<B<10 and 0<f1<10 and 0<B2<10 and 0<f2<10:
                    return 0.0
    else:

            B = params[0]
            #beta = params[1]
            alpha_perp = params[1]
            alpha_par = params[2]
            #f = params[3]
            f = 1.0        
            if 0.8 < alpha_perp < 1.2 and 0.8 < alpha_par < 1.2 and 0.0<B<10 and 0<f<10.0:
                    return 0.0
    return -np.inf



if __name__ == "__main__":

    import sys
    sys.path.append("./")
    #sys.path.insert(0, '/home/merz/workdir/BAOfitter/')
    from analyticBBsolver import LLSQsolver
    import shared

    pardict = ConfigObj('config.ini')

    #Cosmo params
    redshift = float(pardict["z"])
    h = float(pardict["h"])
    n_s = float(pardict["n_s"])
    omb0 = float(pardict["omega0_b"])
    Om0 = float(pardict["omega0_m"])
    sig8 = float(pardict["sigma_8"])



    linearpk = pardict["linearpk"]
    inputpk = pardict["inputpk"]
    window = pardict["window"]
    wideangle = pardict["wideangle"]

    covpath = pardict["covmatrix"]
    outputMC = pardict["outputMC"]


    combined = int(pardict["combined"])
    poles = list(pardict["poles"])
    ell = list(map(int, poles))
    
    deg = list(pardict["degrees"])
    degrees = list(map(int, deg))
    kmin = float(pardict["kmin"])
    kmax = float(pardict["kmax"])
    json = int(pardict["json"])
    convolved = int(pardict["convolve"])

    

    if convolved and 4 in ell:
        from models import modelWl024 as model
        
        
        
    elif not convolved and 4 in ell:
        from models import modelnoWl024 as model
        
        
        
    elif convolved and not 4 in ell:
        from models import modelWl02 as model
        
        
    elif not convolved and not 4 in ell:
        from models import modelnoWl02 as model
    
    smooth = shared.smooth
    print('smooth: ',smooth)

   
    Pkdata = shared.Pkdata
    kobs = shared.kobs
    size = shared.size
    ksize = shared.ksize
    half = shared.half
    km = shared.km

    cov = shared.cov
    covinv = shared.covinv

    if convolved:
        W = shared.W
        M = shared.M
    
    print('poles: ', ell,'redshift: ',redshift)

    
    start = np.array([2.0,1.00,1.00])

    if combined:
        start = np.array([2.0,1.0,1.0,1.0])


    pos0 = start + 1e-4*np.random.randn(8*start.size, start.size)
    nwalkers, ndim = pos0.shape

    print('Running best fit....')
    result = op.minimize(chi2f,start,method='Powell')

    print(result)


    pbf = result.x
    mP = model(pbf,combined)
    Pm= mP.run(km)
    chi2bf = chi2f(pbf)




    np.savetxt(outputMC+'_bf_params.txt',[*pbf,chi2bf])


                                                
    if not combined and 4 in ell:
        np.savetxt(outputMC+'_best_pk.txt',np.column_stack([kobs,Pm[0:ksize],Pm[ksize:2*ksize],Pm[2*kobs.size:3*kobs.size]]))
    elif not combined and not 4 in ell:
        np.savetxt(outputMC+'_best_pk.txt',np.column_stack([kobs,Pm[0:ksize],Pm[ksize:2*ksize]]))

    else:
                np.savetxt(outputMC+'_best_pk.txt',np.column_stack([kobs1,Pm[0:ksize],Pm[ksize:2*ksize],Pm[2*ksize:3*ksize],kobs2,Pm[half:half+ksize],Pm[half+ksize:half+2*ksize],Pm[half+2*ksize:half+3*ksize]]),header='NGC k \t P0 \t P2')




    # Don't forget to clear it in case the file already exists
    filename = outputMC+'.h5'
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # Initialize the sampler
    with Pool() as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlh, pool=pool,backend=backend)
        #sampler = zeus.sampler(nwalkers,ndim,chi2f,pool=pool)
        sampler.run_mcmc(pos0, 5000, progress=True)

    reader = emcee.backends.HDFBackend(outputMC+'.h5')
    samples1 = reader.get_chain(flat=True,discard=200,thin=30)
    B1m = np.mean(samples1[:,0])
    aperm = np.mean(samples1[:,1])
    aparm = np.mean(samples1[:,2])
    #f1m = np.mean(samples1[:,3])
    #B2m = np.mean(samples1[:,3])
    #f2m = np.mean(samples1[:,5])
    paramMC = [B1m,aperm,aparm]

    chi2MC = chi2f(paramMC)
    print(chi2MC)
    np.savetxt(outputMC+'_MC_params.txt',[*paramMC,chi2MC])











