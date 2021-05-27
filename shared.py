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

from analyticBBsolver import LLSQsolver

def prepare_poly_k(ell,convolved):
    if not combined:
        if convolved:
            km = np.linspace(0.001,0.4,endpoint=False,num=400)
            kmodel_vec = np.concatenate([km,km,km])
            kw = np.dot(M,kmodel_vec)
            kw = np.dot(W,kw)
            newkw = np.reshape(kw,(5,40))
            kslice10 = newkw[0][int(kmin/0.01):int(kmax/0.01)]
            kslice12 = newkw[2][int(kmin/0.01):int(kmax/0.01)]
            kslice14 = newkw[4][int(kmin/0.01):int(kmax/0.01)]
                                
        
        if 4 in ell and convolved:
            kwm = [kslice10,kslice12,kslice14]

        elif 4 in ell and not convolved:
            kwm = [kobs,kobs,kobs]
            km = kobs

        elif not 4 in ell and convolved:
            kwm = [kslice10,kslice12]
            
        elif not 4 in ell and not convolved:
            kwm = [kobs,kobs]
            km = kobs

                                                                                                                                                                                                                                                                                                                      

    if combined:
            if convolved:
                km1 = np.linspace(0.001,0.4,endpoint=False,num=400)
                km2 = np.linspace(0.001,0.4,endpoint=False,num=400)
                modelhalf = km1.size
                km = np.concatenate([km1,km2])
                modelsize = km.size
                kmodel_vec = np.concatenate([km,km,km])

                kw = np.matmul(M,kmodel_vec)
                kw = np.matmul(W,kw)
                newkw = np.reshape(kw,(10,40))

                kslice10 = newkw[0][2:23]
                kslice12 = newkw[2][2:23]
                kslice14 = newkw[4][2:23]

                kslice20 = newkw[5][2:23]
                kslice22 = newkw[7][2:23]
                kslice24 = newkw[9][2:23]                                                                

            if 4 in ell and convolved:
                kwm = [kslice10,kslice12,kslice14,kslice20,kslice22,kslice24]

            elif 4 in ell and not convolved:
                kwm = [kobs[0:ksize],kobs[0:ksize],kobs[0:ksize],kobs[ksize:],kobs[ksize:],kobs[ksize:]]
                km = kobs

            elif not 4 in ell and convolved:
                kwm = [kslice10,kslice12,kslice20,kslice22]

            elif not 4 in ell and not convolved:
                kwm = [kobs[0:ksize],kobs[0:ksize],kobs[ksize:],kobs[ksize:]]
                km = kobs
            
            
            
    return kwm,km




def Psmfitfunopt(k,a1,a2,a3,a4,a5):
    Psmfitpre = Psmlinfunc(ktemp) + a1/ktemp**3 + a2/ktemp**2 
    + a3/ktemp + a4 + a5*ktemp
    #Psmfit = np.interp(k,ktemp[2900:5900],Psmfitpre)
    Pspl = IUS(ktemp,Psmfitpre)
    
    #return Psmfit
    return Pspl(k)



def Olin(k):
    #a1 = asm1
    #a2 = asm2
    #a3 = asm3
    #a4 = asm4
    #a5 = asm5
    Olin = Plinfunc(k)/Psmfit(k)
    return Olin




def Psmfit(k):
        #Psmfit = np.interp(k,ktemp[2900:5900],Psmfitopt)
        Psmfit = IUS(ktemp,Psmfitopt)
        #return Psmfit
        return Psmfit(k)

def Legendre(el):
    if el == 0:
        L = 1
    elif el ==2:
        L = 0.5 *(3*muobs**2-1)
    elif el ==4:
        L = 1.0/8*(35*muobs**4 - 30*muobs**2 +3)
    return L




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
inputpk2 = pardict["inputpk2"]
window = pardict["window"]
wideangle = pardict["wideangle"]

covpath = pardict["covmatrix"]
outputMC = pardict["outputMC"]


combined = int(pardict["combined"])
poles = list(pardict["poles"])
ell = list(map(int, poles))
ell = np.asarray(ell)

deg = list(pardict["degrees"])
degrees = list(map(int, deg))
kmin = float(pardict["kmin"])
kmax = float(pardict["kmax"])
dk = float(pardict["dk"])
covstart = float(pardict["covstart"])
json = int(pardict["json"])
convolved = int(pardict["convolve"])
smooth = int(pardict["smooth"])

if json:
    r = ConvolvedFFTPower.load(inputpk)
    poles = r.poles
    shot = poles.attrs['shotnoise']

    P0dat = poles['power_0'].real-shot
    P2dat = poles['power_2'].real
    P4dat = poles['power_4'].real
    kdat = poles['k']

else:
    r = np.loadtxt(inputpk)
    kdat = r[:,0]
    P0dat = r[:,1]
    P2dat = r[:,2]
    P4dat = r[:,3]

valid = (kdat>kmin) & (kdat<kmax)
kobs = kdat[valid]
ksize = kobs.size
P0dat = P0dat[valid]
P2dat = P2dat[valid]
P4dat = P4dat[valid]

if 4 in ell:
        Pkdata = np.concatenate([P0dat,P2dat,P4dat])

else:
        Pkdata = np.concatenate([P0dat,P2dat])

if json and combined:
    r1 = ConvolvedFFTPower.load(inputpk)
    poles1 = r1.poles
    shot1 = poles1.attrs['shotnoise']
    P0dat1 = poles1['power_0'].real-shot1
    P2dat1 = poles1['power_2'].real
    P4dat1 = poles1['power_4'].real
    kdat1 = poles1['k']
    valid1 = (kdat1>0.02) & (kdat1<0.23)
    kobs1 = kdat1[valid1]
    ksize = kobs1.size
    P0dat1 = P0dat1[valid1]
    P2dat1 = P2dat1[valid1]
    P4dat1 = P4dat1[valid1]



    r2 = ConvolvedFFTPower.load(inputpk2)
    poles2 = r2.poles
    shot2 = poles2.attrs['shotnoise']
    P0dat2 = poles2['power_0'].real-shot2
    P2dat2 = poles2['power_2'].real
    P4dat2 = poles2['power_4'].real
    kdat2 = poles2['k']
    valid2 = (kdat2>0.02) & (kdat2<0.23)
    kobs2 = kdat2[valid2]
    P0dat2 = P0dat2[valid2]
    P2dat2 = P2dat2[valid2]
    P4dat2 = P4dat2[valid2]
    kobs = np.concatenate([kobs1,kobs2])
    #Pkdata = np.concatenate([P0dat1,P2dat1,P4dat1,P0dat2,P2dat2,P4dat2])
    
    if 4 in ell:
        Pkdata = np.concatenate([P0dat1,P2dat1,P4dat1,P0dat2,P2dat2,P4dat2])
    else:
        Pkdata = np.concatenate([P0dat1,P2dat1,P0dat2,P2dat2])

size = Pkdata.size
print('size of kobs ',ksize)
half = int(size/2)





covfull = np.loadtxt(covpath)
cov_start = covstart

if not combined:
    nlines = int(covfull.shape[0]/3)
    fac=1
else:
    nlines = int(covfull.shape[0]/6)
    fac=2
    
lowerind = round((kmin-cov_start)/dk)

upperind = round((kmax-cov_start)/dk)


cov = np.zeros((ell.size*ksize*fac,ell.size*ksize*fac))

for i in range(0,ell.size*fac):
    for j in range(0,ell.size*fac):
        cov[i*ksize:(i+1)*ksize,j*ksize:(j+1)*ksize] = covfull[i*nlines+lowerind:i*nlines+upperind,j*nlines+lowerind:j*nlines+upperind]


print('covariance shape',cov.shape)
covinv = inv(cov)


temp = np.loadtxt(linearpk)
#ktemp = temp[0]
#Plintemp = temp[1]
ktemp = temp[:,0]
Plintemp = temp[:,1]


cosmo = cosmology.Cosmology(h=h,Omega0_b=omb0/h**2,n_s=n_s).match(Omega0_m=Om0)  
new_cosmo = cosmo.match(sigma8=sig8)
if sig8 == -1:
    new_cosmo = cosmo


Plinfunc =  IUS(ktemp,Plintemp)
#Plinfunc = cosmology.LinearPower(new_cosmo, redshift=redshift, transfer='CLASS')
Psmlinfunc = cosmology.LinearPower(new_cosmo, redshift=redshift, transfer='NoWiggleEisensteinHu')
    
popt,pcov = curve_fit(Psmfitfunopt,ktemp,Plinfunc(ktemp))
asm1 = popt[0]
asm2= popt[1]
asm3 = popt[2]
asm4 = popt[3]
asm5 = popt[4]

Psmfitopt = Psmfitfunopt(ktemp,asm1,asm2,asm3,asm4,asm5)
        
        
    
muobs = np.linspace(-1,1,100)
sigpar = 8.
sigperp = 3.


if smooth:
    sigpar = 100.
    sigperp = 100.

#print(sigpar,sigperp)

sigs = 4.0

z = redshift
Omv0 = 1-Om0
Omz = Om0*(1+z)**3/(Om0*(1+z)**3 + .69)
f = Omz**0.55

#print('calculated f: ', f)

    
L0 = Legendre(0)
L2 = Legendre(2)
L4 = Legendre(4)


if convolved:
    Wfile = window
    Mfile = wideangle
    W = np.loadtxt(Wfile)
    M = np.loadtxt(Mfile)


kbb,km = prepare_poly_k(ell,convolved)
modelhalf = int(km.size/2)
modelsize = int(km.size)

solver = LLSQsolver(degrees,ell,cov,kbb,combined)
        
        
        
        
        
        
