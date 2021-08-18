"""
Simpleinfo: simple calculation of binned information theoretic quantities

"""

import numpy as np
import scipy as sp
import scipy.stats
from numpy.random import default_rng
rng = default_rng()


def entropy(p):
    """ entropy(p)
    Entropy of a probability distribution vector
    """
    H = -np.sum(p[np.nonzero(p)] * np.log2(p[np.nonzero(p)])) # -p*log2(p)
    return H


def calcinfo(x, xb, y, yb, bias=True, calc_p=False, beta=0.):
    """ (I, p) = calcing(x, xb, y, yb, calc_p=False, beta)
    Calculate mutual information and p value (if calc_p=True) between 
    discrete data sets x and y.
    I = MI( X; Y )
    x should take values in [0 xb-1]
    y should take values in [0 yb-1]
    beta is additive/Laplace smoothing (0.5 = KT estimator)
    Outputs:
    I : MI I(X; Y)
    p : p-value for MU (analytic from chi-square distribution)
    """
    
    # check input
    Ntrl = len(x)
    if Ntrl != len(y):
        raise ValueError('calcinfo: Number of trials must match.')
    
    # calculate the probability distribution vector from 
    # a vector of integer value trials/samples
    xedges = np.arange(-0.5, xb+0.6);
    yedges = np.arange(-0.5, yb+0.6);
    counts, xe, ye = np.histogram2d(x,y,[xedges,yedges])
    Pxy = (counts+beta) / (float(Ntrl)+beta*len(counts))
    Py = np.sum(Pxy, axis = 0, keepdims = True)
    Px = np.sum(Pxy, axis = 1, keepdims = True)
    
    # calculate MI = H(X) + H(Y) - H(X,Y)
    Inobc = entropy(Px) + entropy(Py) - entropy(Pxy)

    if bias:
        I = Inobc - mmbiasinfo(xb, yb, Ntrl)
    else:
        I = Inobc
    
    if calc_p:
        p = 1 - sp.stats.chi2.cdf(2*Ntrl*np.log(2)*Inobc, (xb-1)*(yb-1))
        return (I, p)
    else:        
        return I

def mmbiasinfo(Nx, Ny, Ntrl):
    """bias = mmbiasinfo(Nx, Ny, Ntrl)
    Miller-Madow bias estimate for subtraction from uncorrected binned
    mutual information values
    Nx - number of bins for first variable
    Ny - number of bins for second variable
    Ntrl - number of trials
    """
    return (Nx-1)*(Ny-1) / (2*Ntrl*np.log(2));


def calcpmi(x, xb, y, yb, weighted=False, calc_p=False, beta=0.):
    """ (I, p) = calcing(x, xb, y, yb, calc_p=False, beta)
    Calculate pointwise mutual information and p value (if calc_p=True) between 
    discrete data sets x and y.
    (I, PMI) = MI( X; Y )
    x should take values in [0 xb-1]
    y should take values in [0 yb-1]
    beta is additive/Laplace smoothing (0.5 = KT estimator)
    Outputs:
    I : MI I(X; Y)
    PMI : matrix of pointwise MI values
    p : p-value for MU (analytic from chi-square distribution)
    """

    Ntrl = len(x)
    if Ntrl != len(y):
        raise ValueError('calcinfo: Number of trials must match.')

    # calculate the probability distribution vector from 
    # a vector of integer value trials/samples
    xedges = np.arange(-0.5, xb+0.5);
    yedges = np.arange(-0.5, yb+0.5);
    counts, xe, ye = np.histogram2d(x,y,[xedges,yedges])
    Pxy = (counts+beta) / (float(Ntrl)+beta*len(counts))
    Py = np.sum(Pxy, axis = 0, keepdims = True)
    Px = np.sum(Pxy, axis = 1, keepdims = True)
    Pxyind = Py*Px
    
    PMI = np.zeros_like(Pxy)
    idx = Pxy>0
    PMI[idx] = np.log2(Pxy[idx]) - np.log2(Pxyind[idx])
    PMI[Pxy==0] = 0

    summand = Pxy*PMI
    I = np.sum(summand)
    
    if weighted:
        PMI = summand
    
    if calc_p:
        p = 1 - sp.stats.chi2.cdf(2*Ntrl*np.log(2)*I, (xb-1)*(yb-1))
        return (I, PMI, p)
    else:        
        return (I, PMI)


def calcsmi(x, xb, y, yb, weighted=False, beta=0.):
    """ (I, p) = calcing(x, xb, y, yb, calc_p=False, beta)
    Calculate samplewise mutual information between 
    discrete data sets x and y.
    I = MI( X; Y )
    x should take values in [0 xb-1]
    y should take values in [0 yb-1]
    beta is additive/Laplace smoothing (0.5 = KT estimator)
    Outputs:
    I : MI I(X; Y)
    p : p-value for MU (analytic from chi-square distribution)
    """

    Ntrl = len(x)
    if Ntrl != len(y):
        raise ValueError('calcinfo: Number of trials must match.')

    (I, PMI) = calcpmi(x, xb, y, yb, weighted=weighted, beta=beta)

    SMI = PMI[x,y]
    
    return (I, SMI)


def calccmi(x, xb, y, yb, z, zb, bias=True, calc_p=False, beta=0.):
    """[I p] = calccmi(x, xb, y, yb, z, zb)
    calculate conditional mutual information and p value between
    discrete data sets x and y, conditioning out z
    I = MI( X ; Y | Z )
    x should take values in [0 xb-1]
    y should take values in [0 yb-1]
    z should take values in [0 zb-1]
    """
    # check input
    Ntrl = len(x)
    if (Ntrl != len(y)) or (Ntrl != len(z)):
        raise ValueError('calccmi: Number of trials must match.')
    
    # calculate the probability distribution vector from 
    # a vector of integer value trials/samples
    xedges = np.arange(-0.5, xb+0.5);
    yedges = np.arange(-0.5, yb+0.5);
    zedges = np.arange(-0.5, zb+0.5);
    counts, e = np.histogramdd([x,y,z],[xedges,yedges,zedges])
    Pxyz = (counts+beta) / (float(Ntrl)+beta*len(counts))
    Pxz = np.sum(Pxyz, axis=1, keepdims=True)
    Pyz = np.sum(Pxyz, axis=0, keepdims=True)
    Pz = np.sum(Pxyz, axis=1, keepdims=True).sum(axis=0, keepdims=True)
    
    # calculate CMI(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z)
    Inobc = entropy(Pxz) + entropy(Pyz) - entropy(Pxyz) - entropy(Pz)

    if bias:
        I = Inobc - mmbiascmi(xb, yb, zb, Ntrl)
    else:
        I = Inobc
    
    if calc_p:
        p = 1 - sp.stats.chi2.cdf(2*Ntrl*np.log(2)*Inobc, zb*(xb-1)*(yb-1))
        return (I, p)
    else:        
        return I


def mmbiascmi(Nx, Ny, Nz, Ntrl):
    """bias = mmbiasinfo(Nx, Ny, Ntrl)
    Miller-Madow bias estimate for subtraction from uncorrected binned
    mutual information values
    Nx - number of bins for first variable
    Ny - number of bins for second variable
    Ntrl - number of trials
    """
    return Nz*(Nx-1)*(Ny-1) / (2*Ntrl*np.log(2));


#
# Utility functions
# 

def calcinfoperm(x, xb, y, yb, Nperm, bias=True, beta=0.):
    """calcinfoperm
    Calculate samples from the permutation null that X and Y are independent by 
    shuffling the relationship
    """
    Iperm = np.zeros(Nperm)
    Ntrl = len(x)
    for pi in range(Nperm):
        idx = rng.permutation(Ntrl)
        Iperm[pi] = calcinfo(x[idx], xb, y, yb, bias=False, beta=beta)

    if bias:
        Iperm = Iperm - mmbiasinfo(xb, yb, Ntrl)

    return Iperm

def eqpopbin(x, nb, return_edges=False):
    """eqpopbin(x, nb)
    Bin a sequence of continuou values (x) into nb discrete categories which 
    are approximately equally occupied
    Outputs:
    xb: x binned (integer values 0:nb-1)
    edges: nb-1 bin edges
    """
    
    sx = np.sort(x)
    N = len(sx)

    # determin bin edges
    numel_bin = np.floor(N / nb)
    r = N - (numel_bin*nb)
    
    edges_quantile = np.linspace(0, 100, nb+1)
    edges = np.asarray(np.percentile(x, edges_quantile))

    xb = np.digitize(x, edges[1:], right=True)

    if return_edges:
        return xb, edges
    else:
        return xb


def rebin(x, nb):
    """rebin(x, nb) - rebin an integer sequence
    Rebin an already discretised sequence (eg of intger values) into m levels but iteratively
    merging smallest neighbouring bins 
    """

    # test for positive integer input
    if np.any(np.mod(x,1)) or x.min()<0:
        raise ValueError('Input must be positive integers')

    if x.max() < nb:
        # nothing to do 
        xrb = x
        return xrb

    counts = list(np.bincount(x))
    labels = list(np.arange(len(counts)))
    Nbins = len(counts)
    xrb = x.copy()

    def merge_bins(a,b):
        # match matlab scoping with nonlocal keyword
        nonlocal xrb, counts, labels, Nbins
        # merge bin a into bin b
        counts[b] = counts[b] + counts[a]
        xrb[xrb==labels[a]] = labels[b]
        del labels[a]
        del counts[a]
        Nbins = Nbins - 1

    while Nbins>nb:
        cidx = np.array(counts).argsort()
        # smallest bin
        si = cidx[0]
        # if at the edge can only merge one way
        if si==0:
            merge_bins(si,1)
        elif si==Nbins-1:
            merge_bins(si,si-1)
        else: # merge to the smallest neighbour
            target = np.array([si-1, si+1])
            ti = np.array([counts[si-1], counts[si+1]]).argmin()
            merge_bins(si, target[ti])

    # relabel
    for i in range(Nbins):
        if i != labels[i]:
            xrb[xrb==labels[i]] = i

    return xrb

        




