""" 
Created on Fri Oct  6 14:54:30 2023

@author: ThronC

email:  thron@tamuct.edu

You are welcome to contact me if you have questions!


This code implements the bias corrections described in:
    
    Thron, C., & Jafari, F. (2024). Correcting Scale Distortion in RNA Sequencing Data. 
    
    (Submitted to  BMC)
    
Abbreviations:
LL:  local leveled
NL: weighted least squares

"""


####################
#
#     Initialization
#
####################


## necessary imports

import numpy as np
import random
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from correcting_scale_distortions_functions import readDatafn,ZLS_LLfn2,ZLS_NLfn2,ZLS_NLfn3
import pickle as pkl
from correcting_scale_distortion_plot_routines import oneTimePlots,\
          perDivisionPlots


## File IO

# File read parameters
readData = True
filename = "laml_tcga_pan_can_atlas_2018_data_mrna_seq_v2_rsem.csv"
delim = ','
geneSymIx=0# Column with gene symbol for IMM
geneSymIx = 1# For bladder

# File write parameters
paramsFile = "test.pkl"
#Flag whether to write or not
pickleDump = False
# Notify of where the data will be stored (in case the user will overwrite another file by mistake)
if pickleDump:
    print("@@@NOTE@@@ pickle file name = "+paramsFile)


## Data transform parameters

# Convert data to TPM or not
convertToTPM = False

# Which transform  (log, VST, or none)
transformName = "log"
# VST constants
VST_a0=0.002
VST_a1=5

# Minimum cutoff (remove all genes with mean value less than this)
minTransformedMean = 3.0
logAdd = 0.25 # Additive constant for log transform


## Scale correction parameters

avgWidth = 50 # Width for averaging before fits
degreeLL=3 # Degree of LL LS fit  NOTE verified degree=2 is the same as degree=5
degreeNL=3# Degree of NL fit
methodLabels = ['TPM','TPM-shifted','LL','Nonlinear']


## Test parameters

scrambleGenes = False# If true, scrambles genes before analysis (for comparison)

# Do single-patient comparisons
singlePatient = True

## Simulation parameters:
# - nPatDiv:  number of pop divisions
nPatDiv = 5
# - nSubPop:  number in subpopulation
subpopList = [10,50]
if singlePatient:
    subpopList = [1]# for single-patient
nSubpopList =len(subpopList)
# Spike levels (multiplicative) 
spikeLevels = [1.2,1.4,1.8]
if singlePatient:
    spikeLevels = [1.5,2.0, 3.0]
# Add level 1
spikeLevels=np.array([1.]+spikeLevels)
nSpikeLevel=len(spikeLevels)
spikeLevelString = ["Unenhanced"]
for js in range(1,nSpikeLevel):
    spikeLevelString = spikeLevelString \
        + [str(int((spikeLevels[js]-.999)*100))+"% enhanced"]

# Number of convolutions (corresponds to number of genes tested)
nConvs = 8

## Plot flag variables
#block shift plot (original diagnostic)
plotBlockShift = True
# Whether to plot Gauss avg. in addition to block avg.
plotGaussAvg = False 
# Plot gene-gene correlations
plotCorr = False
# Whether to include Shuffled in correlation plot
plotShuffleCor = False
# Plot distributions of mean and caterpillar plot 
plotDists = False
# Plot nonlinear scale changes for selected patients
plotNonlin = False
# show characteristics of NL adjustment
plotAdjusted = False
# Include Gauss data with same mean and std for plotDivision plots
includeGauss = False

plotDetectionBySubpopSizeFixedDetection = False
plotByNgenesTested = False

### Display parameters

# plotBlockShift parameters
nPatBlock = 10 # Number of patients to display with block avg.
blockSize = 800 # Block size for plotting running average

# plotCor parameters
corrAll = False# include all genes in correlation
corrLim = [4,5]# if do not correlate all, range of mean exp levels  
nGcorr = 1000# Number of genes to correlate if corrAll = False
corrHistBins = np.linspace(-1,1,401)# Bins for histogram

# plotNonlin parameters
nonlinPatNo = 10 # Number of nonlinear scale changes to display

# Necessary if include gauss distribution in plot
if includeGauss:
    methodLabels.append("Gauss")
nMethod = len(methodLabels)

plotSubpops = np.array(subpopList) # which pop. divisions to plot
selectSpikeLevel = 1 # Select which spike level for plotting

# Complementary cdf plots (1-time)
# Lowest complementary cdf for plotting ccdfs
ccdfLim = 0.05# @@@ currently unused
# Highest false positive probability in single-gene ROC plot
ROClim = 0.01

# Params for plots for selected subpops
# ROC lim for convolutional plots
ROClimConv = 0.0001
if singlePatient:
    ROClimConv = 0.01
nROC = 5

coCDF_Tlevels = np.linspace(2.0,3.0,5)

# Distribution of detection plots
tStatLevels = np.array([3.0,3.3,3.6])
tWidth= 800 # width for average
tStep = 800 # step between plot points


# save all parameters to pickle file
listOfParams = [convertToTPM,transformName,VST_a0,VST_a1,minTransformedMean,logAdd,avgWidth,degreeLL,degreeNL,nPatDiv,subpopList,spikeLevels,selectSpikeLevel,ccdfLim,ROClim,ROClimConv]

    
### Read in data
if readData:
    print("Begin reading data")
    nGene,geneSymbol,readMx = readDatafn(filename,delim,geneSymIx)


### Create XLs (XL with spike levels)
# We spike using addition on log transformed instead of multiplication so that spiking the subpopulation has a known effect on the mean

# Convert to TPM if selected.
if convertToTPM:
    readMean = np.mean(readMx,axis=1)
    readMx = readMx/readMean.reshape((-1,1))*1E6

plotTransformed=True# If data is transformed plot shift instead of ratio
if transformName == "log":
    print("Begin log transform")
    # Perform log transform
    Xinit = np.log2( logAdd + readMx)  
elif transformName == "VST":
    print("Begin VST transform")
    # VST transformed data (transformation from bioconductor.org)
    a0 =VST_a0
    a1 = VST_a1
    xTmp = readMx*a0
    Xinit = (np.log(1 + 2*xTmp+a1 
                        + 2*np.sqrt(xTmp*(1+xTmp+a1)))-np.log(4*a0)
              )/np.log(2) 
else:# Use untransformed data
    Xinit = np.copy(readMx)
    plotTransformed = True

# Select genes with mean expression level above minimum
data = Xinit[:,np.mean(Xinit,axis=0)>minTransformedMean]   

# Order genes according to mean expression level
tmp = np.mean(data,axis=0)
ix = np.argsort(tmp)
data = data[:,ix]

# Set parameters
nPat,nGene = data.shape
minDataVal = np.min(data)
maxDataVal = np.max(data)
# Reset these numbers if too large
nPatBlock = min(nPat,nPatBlock)
nonlinPatNo = min(nPat,nonlinPatNo)

# Scramble genes (optional)
if scrambleGenes:
    print("Begin scramble genes")
    # Create an array of random indices for shuffling
    for jc in range(nGene):
       rIx  = np.random.permutation(np.arange(nPat))
       data[:,jc] = data[rIx,jc]



# Perform additive spike (similar to multiplicative)
# Do all spikes at once by adding a dimension
print("Begin spike genes")
XLnoSpike = data.reshape((1,nPat,nGene))
log2SpikeMult = np.log2(spikeLevels).reshape((-1,1))
XLs = log2SpikeMult.reshape((-1,1,1)) + XLnoSpike


# Create XLSs--corrected for overall shift (with spike levels)
# NOTE the shift correction is the same for spiked, because it depends on row means over patients. Spiking a small number of genes will not change the patients' row means.
# Use non-spiked to reset rows
rowMeansVec = np.mean(XLs[0,:,:],axis=1).reshape((1,-1,1))
totMean = np.mean(rowMeansVec)
# Reset all rows
XLSs = XLs-rowMeansVec + totMean



'''
# Compute XLS_NL (unspiked and spiked)
# NOTE  we don't need to rerun the models for each population split, because the correction for a given gene for a given patient the correction depends only on the gene level for that patient, and not on the other patients.'''

print("Begin nonlinear transform")    
# Y for NL fit 
Y_NL = np.mean(XLs[0],axis=0)
# x for NL fit
X_NL = XLs
# std to create weights for LS fit
std_NL = np.std(XLs[0],axis=0)
# Call nonlin function
modelsNL,XLS_NLs = ZLS_NLfn3(Y_NL,X_NL,std_NL,avgWidth,degreeNL)
# (Several false starts for nonlinear)
#    coefsNL, XLS_NLs = ZLS_NLfn2(LHS,XLs,XLS_LLs,stds,avgWidth,degreeNL,normalizeNL)
#    coefsNL, XLS_NLs = ZLS_WLSfn2(LHS,XLs,stds,avgWidth,degreeLL)
#    coefsNL, XLS_NLs = ZLS_LLfn2a(xs,XLs,stds,avgWidth,degreeLL)


# Spiked Gaussian data with same mean and std
if includeGauss:
    XLgauss = np.random.normal(size=(1,nPat,nGene))
    XLgauss = XLgauss*np.std(XLnoSpike,axis=1) \
                 + np.mean(XLnoSpike,axis=1)
    XLgauss = log2SpikeMult.reshape((-1,1,1)) + XLgauss

# Variables for subpop calculations
subPopDependentDetectionMean = np.zeros((nSubpopList,nMethod,nConvs))
subPopDependentDetectionXstd = np.zeros((nSubpopList,nMethod,nConvs))
subPopDependentDetectionYstd = np.zeros((nSubpopList,nMethod,nConvs))

# Needed for LL calc (but only need to compute once)
unspikedMean = np.mean(XLSs[0],axis=0).reshape((1,-1))

'''
Begin loop over subpopulation divisions:
   The following calculations require LL, which is computed differently for different population divisions
'''
for jp,nSubpop in enumerate(subpopList):
    plotThisSubpop = (np.sum(nSubpop==plotSubpops)>0)
    print("Subpop size",nSubpop)

    '''
    Local leveling (LL)
    
    The LL models work also for spiked, because the LL correction depends on a LS fit for lots of genes spiking a small number of genes will not change the LS fit. However, when doing the correction for a particular patient for a particular division, the correction depends on both the model and the mean gene expression level. Thus the average mean gene expression level in the 2-pop case must be evaluated properly. If additive spiking is used, then the effect on the mean is independent of the particular subpopulation chosen.
    '''
    # Find means for spiked also for this population split size
    print("begin LL")
    Y_LL = unspikedMean + log2SpikeMult*nSubpop/nPat
    # Find std's assumes independence
    # (Do not compute std's for spiked, because we're not using these as fitting variables)
    stdsLL = np.std(XLs[0],axis=0)   
    # Call LL function
    modelsLL, XLS_LLs = ZLS_LLfn2(Y_LL,XLs,stdsLL,avgWidth,degreeLL)
    
    # Pack everything into an array with dimension:
    # dimension: nMethods, nSpikeLevel,nPat,nGene 
    if includeGauss:
        arrayOfXs = np.array([XLs,XLSs,XLS_LLs,XLS_NLs,XLgauss])
    else:
        arrayOfXs = np.array([XLs,XLSs,XLS_LLs,XLS_NLs])


    ''' 
    Begin loop over specific divisions for this subpop size. This is to calculate division-specific statistics
    '''
    
    ## Prepare arrays for loop
    # Compute all squares to save time. Doesn't change for different divisions.
    arrayOfXsqs = arrayOfXs**2
    
    # Array that stores spiked subpops
    ixSpiked = np.zeros((nPatDiv,nSubpop))
    
    # Array that stores all t statistics
    arrayOfTstats = np.zeros((nPatDiv,nMethod,nSpikeLevel,nGene))
    
    ## Loop begins--choose random division of population
    for jd in range(nPatDiv):
        print("Patient division number",jd)
        # Select indices for subpop
        ix1 = np.random.choice(nPat, size=nSubpop, replace=False) # Choose random sample
        # Store indices
        ixSpiked[jd,:] = ix1
        # Create boolean version of subpop selection
        ixBool = 0*np.arange(nPat)
        ixBool[ix1] = 1
        # Create boolean for base pop selection
        ix0 = np.where(ixBool==0)
        
        
        # To compute t statistics, need means and variances of both pops
        # For subpop 0, compute unspiked mean and variance
        arrayOfMeans0 = np.mean(arrayOfXs[:,0,ix0,:],axis=2)
        arrayOfVars0 = (np.mean(arrayOfXsqs[:,0,ix0,:],axis=2) - arrayOfMeans0**2)
        # For subpop 1, compute spiked means and variances
        arrayOfMeans1 = np.mean(arrayOfXs[:,:,ix1,:],axis=2)
        arrayOfVars1 = (np.mean(arrayOfXsqs[:,:,ix1,:],axis=2)-arrayOfMeans1**2)
        
        ns0 = nPat - nSubpop
        ns1 = nSubpop
        # Combine to obtain all t statistics
        # Recall dimension of t statistics--
        #    nPatDiv x nMethod x nSpikeLevel x nGene
        # Compute t statistics for this division.
        # Assumes same variance
        arrayOfMeansDiff = (arrayOfMeans1-arrayOfMeans0)
        arrayOfPooledVars = np.sqrt((ns0*arrayOfVars0+ns1*arrayOfVars1)/(nPat-2))
        denom = arrayOfPooledVars*np.sqrt(1/ns0 + 1/ns1)
        arrayOfTstats[jd,:,:,:] = np.squeeze(arrayOfMeansDiff/denom).reshape(nMethod,-1,nGene)
        
    
    # Indices:  Method, spike level, patient,gene
    # Xtest:  unspiked
    Xtest = arrayOfXs[:,0,:,:]
    XtestMean = np.mean(Xtest,axis=1)
    ixSort = np.argsort(XtestMean[0,:])
    Xtest = Xtest[:,:,ixSort]
    
    XtestMean = np.mean(Xtest,axis=1)
    XtestVar = np.var(Xtest,axis=1)
    
    # Prepare for one-time plots
    plotDistParams =\
      plotDists,minDataVal,maxDataVal,data,transformName,nGene
    plotNonlinParams=\
        plotNonlin,nPat,nonlinPatNo,minTransformedMean,\
            maxDataVal,modelsNL
    plotBlockShiftParams =\
        plotBlockShift,plotTransformed,nPat,nPatBlock,arrayOfXs,blockSize,\
        nGene,methodLabels
    plotGaussAvgParams =\
        plotGaussAvg,arrayOfXs,nMethod,\
            nPatBlock,nGene,transformName
    plotCorrParams =\
        plotCorr,corrAll,unspikedMean,corrLim,nGcorr,nGene,corrHistBins,nMethod,includeGauss,plotShuffleCor,methodLabels,arrayOfXs
        
    plotAdjustedParams=\
        plotAdjusted,XtestMean,methodLabels,XtestVar
    
    # Call one time plots
    print("1-time plots begin")
    oneTimePlots(plotDistParams,plotNonlinParams,\
                     plotBlockShiftParams,plotGaussAvgParams,\
                         plotCorrParams,plotAdjustedParams)

    
    # Plot of ROC curve, with error bars
    # Find quantiles for unspiked simultaneously for all divisions:  np.quantile(array, values)
    print("Begin ROC plot")
    qLevels = np.linspace(1,1-ROClim,nROC)
    nQlevel = len(qLevels)
    arrayOfZquantiles =np.quantile(arrayOfTstats[:,:,0,:],qLevels,axis=2).reshape(nQlevel,nPatDiv,nMethod,1,1)
    ROClevels = np.zeros((nQlevel,nPatDiv,nMethod,nSpikeLevel-1))
    for jq in range(nQlevel):
        ROClevels[jq,:,:,:] = np.sum(arrayOfTstats[:,:,1:,:] > arrayOfZquantiles[jq,:,:,:],axis=3)
    ROClevels = ROClevels/nGene
    # Compute mean and standard deviation
    ROCmean = np.mean(ROClevels, axis=1)
    ROCstd = np.std(ROClevels, axis=1)
    
    ROCquan05 = np.quantile(ROClevels,0.05, axis=1)
    ROCquan95 = np.quantile(ROClevels,0.95, axis=1)
    
    if plotThisSubpop and nSpikeLevel>1:    
        # Plot ROC curve summary.
        fig, axes = plt.subplots(ncols=nSpikeLevel-1,nrows=1,  figsize=(6,4))
        if nSpikeLevel==2:
            axes=[axes]
        xVec = 1-qLevels
        # Interval between error bars for plot
        xVecStep=(xVec[1]-xVec[0])*.4/nMethod
        
        for jm in range(nMethod):
            # False positives for this method (spike level 0)
            for js in range(nSpikeLevel-1):
                yVec = np.squeeze(ROCmean[:,jm,js])
                yStd = np.squeeze(ROCstd[:,jm,js])
                yMinusPlus = np.array([np.squeeze(ROCquan05[:,jm,js]),np.squeeze(ROCquan95[:,jm,js])])
                yMinusPlus = np.abs(yMinusPlus-yVec)
                if includeGauss and jm==nMethod-1:
                    axes[js].errorbar(xVec+xVecStep*jm,yVec,yerr=yMinusPlus,color="black",linestyle="dashed",linewidth=1.5)
                else:
                    axes[js].errorbar(xVec+xVecStep*jm,yVec,yerr=yMinusPlus)    
                axes[js].set_title(spikeLevelString[1+js])
                axes[js].set_xlabel("False positive rate")
                axes[js].set_ylabel("True positive rate")
        plt.suptitle( "ROC curves, subpopulation size "+str(subpopList[jp]))
        plt.legend(methodLabels)
        plt.tight_layout()
        plt.show() 
    
    # For given t statistic--
    # For all spiked and unspiked, compute percentage greater than that t statistic
    # For each spike level and method, compute mean and standard deviation of percentages.
    print("Begin coCDF calculation")
    Tlevels = coCDF_Tlevels
    nTlevel = len(Tlevels)
    arrayOfCoCDF = np.zeros((nTlevel,nPatDiv,nMethod,nSpikeLevel))
    for jz in range(nTlevel):
        arrayOfCoCDF[jz,:,:,:] = np.sum(arrayOfTstats > Tlevels[jz],axis=3)
    arrayOfCoCDF = arrayOfCoCDF/nGene
    coCDFmean = np.mean(arrayOfCoCDF,axis=1)
    coCDFstd = np.std(arrayOfCoCDF,axis=1)    
    coCDFquan05 = np.quantile(arrayOfCoCDF,0.05, axis=1)
    coCDFquan95 = np.quantile(arrayOfCoCDF,0.95, axis=1)
    
    if plotThisSubpop:
        print("Begin coCDF plot")
        # Plot complementary cdfs.
        fig, axes = plt.subplots(ncols=nSpikeLevel, nrows=1,  figsize=(8,4))
        xVec = Tlevels
        # Interval between error bars for plot
        xVecStep=(xVec[1]-xVec[0])*.15/nMethod

        
        for jm in range(nMethod):
            # False positives for this method (spike level 0)
            for js in range(nSpikeLevel):
                yVec = np.squeeze(coCDFmean[:,jm,js])
                yStd = np.squeeze(coCDFstd[:,jm,js])
                yMinusPlus = np.array([np.squeeze(coCDFquan05[:,jm,js]),np.squeeze(coCDFquan95[:,jm,js])])
                yMinusPlus = np.abs(yMinusPlus-yVec)
                if includeGauss and jm==nMethod-1:  
                    axes[js].errorbar(xVec+xVecStep*jm,yVec,yerr=yMinusPlus, color = "black", linestyle="dashed",linewidth=1.5)
                else:        
                    axes[js].errorbar(xVec+xVecStep*jm,yVec,yerr=yMinusPlus)
                axes[js].set_title(spikeLevelString[js])
                axes[js].set_xlabel("t statistic")
                axes[js].set_ylabel("Probability")
        plt.suptitle( "Complementary cdfs for subpopulation size "+str(subpopList[jp]))
        plt.legend(methodLabels)
        plt.tight_layout()
        plt.show() 
    
    
    # Compute convolutions
    print("Begin convolution calculation")
    Tlevels2 = np.linspace(-40,40,513)
    nTlevel2 = len(Tlevels2)-1
    arrayOf_pdf = np.zeros((nTlevel2+1,nPatDiv,nMethod,nSpikeLevel))
    for jz in range(nTlevel2+1):
        arrayOf_pdf[jz,:,:,:] = np.sum(arrayOfTstats[:,:,:,:] > Tlevels2[jz],axis=3)
    arrayOf_pdf = arrayOf_pdf[1:,:,:,:] - arrayOf_pdf[:-1,:,:,:]
    arrayOf_pdf = arrayOf_pdf / np.sum(arrayOf_pdf,axis=0,keepdims=True)
        
    arrayOf_convs = np.zeros((nConvs,nTlevel2,nPatDiv,nMethod,nSpikeLevel ))
    for jd in range(nPatDiv):
        for jm in range(nMethod):
            for js in range(nSpikeLevel):
                arrayOf_convs[0,:,jd,jm,js] = arrayOf_pdf[:,jd,jm,js]
                # Ensure probabilities greater than 0
                for jc in range(1,nConvs):
                    arrayOf_convs[jc,:,jd,jm,js] = np.maximum(0,np.convolve(arrayOf_pdf[:,jd,jm,js],arrayOf_convs[jc-1,:,jd,jm,js],mode = 'same'))
                    
        
    print("Begin ROC calculation for conv.")     
    arrayOfCoCDF2 = np.maximum(0, 1 - np.cumsum(arrayOf_convs, axis = 1))
    # interpolate at fixed values
    FPRs = np.linspace(0,ROClimConv,9)
    # dimension array to hold array
    arrayOfROC = np.zeros((nConvs,len(FPRs),nPatDiv,nMethod,nSpikeLevel))
    ixs = np.indices((nConvs,nPatDiv,nMethod,nSpikeLevel)) 
    for jf, FPR in enumerate(FPRs):
        #Find index in unspiked
        # -1 is necessary in case index is last
        FPRix = np.sum(arrayOfCoCDF2[:,:,:,:,:1] >= FPR,axis=1)-1
        # Corresponding index in spiked
        arrayOfROC[:,jf,:,:,:] = arrayOfCoCDF2[ixs[0],FPRix,ixs[1],ixs[2],ixs[3]]
    arrayOfROCmean = np.mean(arrayOfROC,axis=2)
    arrayOfROCstd = np.std(arrayOfROC,axis=2)
    arrayOfROCquan05 = np.quantile(arrayOfROC,0.05, axis=2)
    arrayOfROCquan95 = np.quantile(arrayOfROC,0.95, axis=2)

    arrayOfCoCDFmean = np.mean(arrayOfCoCDF2,axis=2)
    arrayOfCoCDFstd = np.std(arrayOfCoCDF2,axis=2)
    arrayOfCoCDFquan05 = np.quantile(arrayOfCoCDF2,0.05, axis=2)
    arrayOfCoCDFquan95 = np.quantile(arrayOfCoCDF2,0.95, axis=2)

    if plotThisSubpop:
        convArr = np.array([2,4,8])
        
        print("Begin ROC plot for conv.")        
        fig, axes = plt.subplots(ncols=3, nrows=1,  figsize=(6,4))
        dFPR=(FPRs[1]-FPRs[0])*.6/nMethod
        for jpc,jc in enumerate(convArr):
            # Plot complementary cdfs.
            for jm in range(nMethod):
                # xVec = np.squeeze(arrayOfCoCDFmean[jc,:,jm,0])
                # ixs = (xVec < ROClimConv)
                # yVec = np.squeeze(arrayOfCoCDFmean[jc,:,jm,selectSpikeLevel])
                # xStd =  np.squeeze(arrayOfCoCDFstd[jc,:,jm,0])
                # yStd = np.squeeze(arrayOfCoCDFstd[jc,:,jm,selectSpikeLevel])
                # axes[jc//3,jc%3].errorbar(xVec[ixs],yVec[ixs],xerr=xStd[ixs]/np.sqrt(nPatDiv),yerr=yStd[ixs]/np.sqrt(nPatDiv))
                yVec = np.squeeze(arrayOfROCmean[jc-1,:,jm,selectSpikeLevel])
                # Don't use yStd for errobar any more
                yStd = np.squeeze(arrayOfROCstd[jc-1,:,jm,selectSpikeLevel])
                # For errorbar lower & upper limits
                ym = np.squeeze(arrayOfROCquan05[jc-1,:,jm,selectSpikeLevel])
                yM = np.squeeze(arrayOfROCquan95[jc-1,:,jm,selectSpikeLevel])
                yMinusPlus = np.array([ym,yM])
                yMinusPlus = np.abs(yMinusPlus-yVec)
                if includeGauss and jm==nMethod-1:
                    axes[jpc].errorbar(FPRs+dFPR*jm,yVec,yerr=yMinusPlus,color="black",linestyle="dashed",linewidth=1.5)
                else:
                    axes[jpc].errorbar(FPRs+dFPR*jm,yVec,yerr=yMinusPlus)        
                axes[jpc].set_title("Sum of "+str(jc)+" genes")
                axes[jpc].set_xlabel("False positive rate")
                axes[jpc].set_ylabel("True positive rate")
            ## Interpolate values for mean and 2 stds: array is (number of subpop sizes, number of methods, number of convolutions)
            ## this is the old way
            # (Negatives are to make xVec increasing)
            # subPopDependentDetectionMean[jp,jm,jc] = np.interp([-ROClimConv],-xVec,yVec)
            # subPopDependentDetectionXstd[jp,jm,jc] = np.interp([-ROClimConv],-xVec,xStd)
            # subPopDependentDetectionYstd[jp,jm,jc] = np.interp([-ROClimConv],-xVec,yStd)
        plt.suptitle( "ROC for different gene sample sizes, "+spikeLevelString[selectSpikeLevel]+",subpopulation size ="+str(subpopList[jp]))
        plt.legend(methodLabels)
        plt.tight_layout()
        plt.show() 

if plotDetectionBySubpopSizeFixedDetection:
    # plot by subpop size: detection rate at fixed false detection 
    fig, axes = plt.subplots(ncols=3, nrows=(nConvs+2)//3,  figsize=(8,8))
    xVec = np.array(subpopList)
    
    for jc in range(nConvs):
        # Plot complementary cdfs.
        for jm in range(nMethod):
            yVec = np.squeeze(subPopDependentDetectionMean[:,jm,jc])
            xStd =  np.squeeze(subPopDependentDetectionXstd[:,jm,jc])
            yStd = np.squeeze(subPopDependentDetectionYstd[:,jm,jc])
            axes[jc//3,jc%3].errorbar(xVec+jm*1.5,yVec,xerr=xStd/np.sqrt(nPatDiv),yerr=yStd/np.sqrt(nPatDiv))
            axes[jc//3,jc%3].set_title("Sum of "+str(jc+1)+" genes")
            axes[jc//3,jc%3].set_xlabel("Subpop. size (out of "+str(nPat)+")")
            axes[jc//3,jc%3].set_ylabel("Detection probability")
    plt.suptitle( "Detection rate at false detection rate "+str(ROClimConv)+" for different subpop sizes for 1-9 genes tested")
    plt.legend(methodLabels)
    plt.tight_layout()
    plt.show() 

if plotByNgenesTested:
    # plot by number of genes tested for different subpops: detection rate at 0.005
     
    fig, axes = plt.subplots(ncols=2, nrows=(nSubpopList+1)//2,  figsize=(8,8))
    xVec = np.arange(nConvs)+1
    axes = axes.reshape(-1,2)
    
    for jp in range(nSubpopList):
        # Plot complementary cdfs.
        for jm in range(nMethod):
            yVec = np.squeeze(subPopDependentDetectionMean[jp,jm,:])
            xStd =  np.squeeze(subPopDependentDetectionXstd[jp,jm,:])
            yStd = np.squeeze(subPopDependentDetectionYstd[jp,jm,:])
            axes[jp//2,jp%2].errorbar(xVec+jm*.03,yVec,xerr=xStd/np.sqrt(nPatDiv),yerr=yStd/np.sqrt(nPatDiv))       
            axes[jp//2,jp%2].set_title("Subpopulation size "+str(subpopList[jp]))
            axes[jp//2,jp%2].set_xlabel("Number of genes tested")
            axes[jp//2,jp%2].set_xticks(ticks=xVec)
            axes[jp//2,jp%2].set_ylabel("Detection probability")
            
    plt.suptitle( "Detection rate at FPR "+str(ROClimConv)+" vs. no. of genes tested, \n "+spikeLevelString[selectSpikeLevel])
    plt.legend(methodLabels)
    plt.tight_layout()
    plt.show() 

# Plot to show distribution of detections
print("Begin distribution of detections")
fig, axs = plt.subplots(nrows=nSpikeLevel-1, ncols=len(tStatLevels),  figsize=(8,8))

for jt, tLevel in enumerate(tStatLevels):
    for js, sLevel in enumerate(spikeLevels[:-1]):
        outFreqPerGene = np.mean(arrayOfTstats[:,:,js,:] > tLevel, axis = 0)
        cumOutFreq = np.cumsum(outFreqPerGene,axis=1)
        cumOutFreqAvg = (cumOutFreq[:,tWidth:]-cumOutFreq[:,:-tWidth])/tWidth
        
        axs[js,jt].plot(unspikedMean[:,tWidth//2:-tWidth//2:tStep].T,cumOutFreqAvg[:,::tStep].T)
        if sLevel == 1.:
            axs[js,jt].set_title("FPR at t=" + str(tLevel))
        else:           
            axs[js,jt].set_title("TPR, t=" + str(tLevel) + ", "+spikeLevelString[js])
        axs[js,jt].set_xlabel("Mean expression level")
        axs[js,jt].set_ylabel("")
plt.suptitle( "Block averaged outlier detection rate vs. mean expression level")
plt.legend(methodLabels)
plt.tight_layout()
plt.show()   


listOfStds=[]
for jx,Xx in enumerate(arrayOfXs[:4]):
    xTmp = Xx[0]# No spike
    meanTmp = np.mean(xTmp,axis=1)
    meanDiffs = meanTmp -np.mean(meanTmp,axis=0)
    listOfStds.append(np.std(meanTmp))
    print("Standard deviation for ",methodLabels[jx],"=",listOfStds[-1])
    
# Save parameters
listOfParams.append(listOfStds)

if pickleDump:
    with open(paramsFile, 'wb') as file:
        pkl.dump(listOfParams,file)    



