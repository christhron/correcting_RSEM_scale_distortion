# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 17:37:17 2024

@author: USER
"""
import numpy as np
import csv
import statsmodels.api as sm
import scipy as sp

def readDatafn(filename,delim,geneSymIx):
    ## Read data
    with open(filename, "r",newline='') as file:
        reader = csv.reader(file,delimiter=delim)
        for j, line in enumerate(reader):
            if j == 0:
                patNames = line[geneSymIx+1:]
                nPat = len(patNames)
                readMx = np.zeros((100000, nPat))
                geneSymbol = list(range(100000))
            elif 'na' not in [cell.lower() for cell in line]:
                geneSymbol[j-1]=line[geneSymIx]
                readMx[j-1, :] = np.maximum(np.array(line[geneSymIx+1:]).astype(float),0)
    
    ## Extract gene data and gene labels
    nGene = j
    geneSymbol = geneSymbol[:nGene]
    readMx = readMx[:nGene, :].T
    return nGene,geneSymbol,readMx


### Functions


# import numpy as np


'''
Function to apply LL to data
xs -- mean of expression level data
XLs -- expression level data 
'''
def ZLS_LLfn2(xs,XLs,stds,width,degree):
    
    # Data is shifted log data 
    nSpikeLevel,nPat,nGene = np.shape(XLs)
    ix0 = nGene%width
    
    # Data to fit (deviations)
    yData = XLs-xs.reshape(nSpikeLevel,1,nGene)
    # Independent variables for polynomial regression
    # reshaped for block average
    # Independent variables are the same for each patient
    xVars = np.reshape(xs[0,ix0:],(width,-1),order='F')
    xVars = np.mean(xVars,axis=0)

    #Define weights and reshape for block average
    wts = stds[ix0:]**2
    wts = np.reshape(wts,(width,-1),order='F')
    wts = np.sqrt(1/np.mean(wts,axis=0))

    # Initialize arrays for results
    models = [0]*nPat
    fitMx = np.zeros((nSpikeLevel,nPat,nGene)) # To store fits
    
    # Do per-patient fit
    for jd in range(nPat):
        yVar = 1.*yData[0,jd,ix0:]
        yVar = np.reshape(yVar,(width,-1),order='F')
        yVar = np.mean(yVar,axis=0)        
        models[jd] = np.polynomial.Polynomial.fit(xVars, yVar, degree, w=wts)
        
        # Apply fit to all spike levels
        for js in range(nSpikeLevel):
            fitMx[js,jd,:]=XLs[js,jd,:] \
                           - models[jd](xs[js,:]) 
                
    return models, fitMx

'''
Function to apply NL to data
xs -- mean of expression level data
XLs -- expression level data 
'''
def ZLS_NLfn3(xs,XLs,stds,width,degree):
    
    # Data is shifted log data 
    nSpikeLevel,nPat,nGene = np.shape(XLs)
    ix0 = nGene%width
    
    # Data to fit (deviations)
    yVar = 1.*xs[ix0:]
    yVar = np.reshape(yVar,(width,-1),order='F')
    yVar = np.mean(yVar,axis=0)       

    #Define weights and reshape for block average
    wts = stds[ix0:]**2
    wts = np.reshape(wts,(width,-1),order='F')
    wts = np.sqrt(1/np.mean(wts,axis=0))

    # Initialize arrays for results
    models = [0]*nPat
    fitMx = np.zeros((nSpikeLevel,nPat,nGene)) # To store fits
    
    # Do per-patient fit
    for jd in range(nPat):
        xVar = 1.*XLs[0,jd,ix0:]
        xVar = np.reshape(xVar,(width,-1),order='F')
        xVar = np.mean(xVar,axis=0)        
        models[jd] = np.polynomial.Polynomial.fit(xVar, yVar, degree, w=wts)
        
        # Apply fit to all spike levels
        for js in range(nSpikeLevel):
            fitMx[js,jd,:] = models[jd](XLs[js,jd,:]) 
            
    return models, fitMx


'''
Functions used to normalize nonlinear algorithm
@@ However, normalization appears to give worse performance @@
'''
# Function required for constrained optimization
def NLfun(aVec,params):
    yVec,xMx,wVec = params
    fit = np.sum(wVec*(yVec - xMx@aVec)**2)
    return fit

def NLgradfun(aVec,params):
    yVec,xMx,wVec = params
    grad = np.sum(2*xMx.T@(wVec*(xMx@aVec-yVec)))
    return grad.reshape(-1)

def NLconstraintfun(aVec,params):
    xFit,sumVal = params    
    xTmp = xFit-aVec[0]
    disc = aVec[1]**2 + 4*xTmp*aVec[2]
    # Need maximum to avoid sqrt of negative number
    sumFit= 2*xTmp / (np.sqrt(np.maximum(disc,0)) + aVec[1])
    constraint = np.sum(sumFit) - sumVal
    # If min(disc)<0, penalize with increasing penalty
    mindisc = np.min(disc)
    constraint = constraint - 1000*(mindisc<0)*mindisc *np.sign(constraint) 
    return constraint

def ZLS_NLfn2(xs,XLs,XL_LL,stds,width,degree,normalize):
    
    # Data is shifted log data 
    nSpikeLevel,nPat,nGene = np.shape(XLs)
    ix0 = nGene%width
    
    # Data to fit (same as with LL)
    yData = XLs
    
    # Independent variables for polynomial regression
    # reshaped for block average
    # Same as with LL (later will add to quadratic term)
    xVars = np.reshape(xs[ix0:],(width,-1),order='F')
    xVars = np.mean(xVars,axis=0)
    
    # Sum of mean values (used to normalize)
    sumVal = np.sum(xVars)
    
    # Create polynomial features
    powers = np.arange(degree+1).reshape((-1,1))
    xVars =xVars.reshape((1,-1))**powers
    
    #Define weights and reshape for block average
    wts = stds[ix0:]**2
    wts = np.reshape(wts,(width,-1),order='F')
    wts = 1/np.mean(wts,axis=0)

    # Initialize arrays for results
    modelCoefs = np.zeros((nPat,degree+1))
    fitMx = np.zeros((nSpikeLevel,nPat,nGene)) # To store fits
    
    # Do per-patient fit
    for jd in range(nPat):
        # Create dependent variable
        yVar = 1.*yData[0,jd,ix0:]
        yVar = np.reshape(yVar,(width,-1),order='F')
        yVar = np.mean(yVar,axis=0)
        
        # Create term to add to X^2 term
        xDev = (XL_LL[0,jd,:]-xs)**2
        xDev = xDev[ix0:]
        xDev = np.reshape(xDev,(width,-1),order = 'F')
        xDev = np.mean(xDev,axis=0)
        xVars[2,:] = xVars[2,:] + xDev
        # Fit model to get coefficients of fun
        # will use to invert
        model = sm.WLS(yVar, xVars.T, weights = wts).fit()
        aVec=np.array(model.params)

        modelCoefs[jd,:] = aVec
        
        # Apply fit to all spike levels
        for js in range(nSpikeLevel):
            # Obtain expression level for this spike level  
            xMeas = XLs[js,jd,:] 
            xTmp = xMeas -aVec[0]
            nIter = 3
            aDerivVec = np.arange(1,degree+1)*aVec[1:]
            xEst = XL_LL[0,jd,:]
            for ji in range(nIter):
                # Create polynomial features
                xEstPowers = xEst.reshape((1,-1))**powers
                
                fVal = aVec@xEstPowers - xMeas
                fDerivVal = aDerivVec@xEstPowers[:-1,:]
                xEst = xEst - fVal/fDerivVal
                
            fitMx[js,jd,:] = xEst
    
    return modelCoefs, fitMx

