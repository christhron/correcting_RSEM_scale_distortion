# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 07:15:56 2024

@author: USER
"""
# necessary imports
import numpy as np
import random
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns


def oneTimePlots(plotDistParams,plotNonlinParams,\
                 plotBlockShiftParams,plotGaussAvgParams,\
                     plotCorrParams,plotAdjustedParams):
    plotDists,minDataVal,maxDataVal,data,transformName,nGene\
        = plotDistParams
    plotNonlin,nPat,nonlinPatNo,minTransformedMean,\
    maxDataVal,modelsNL\
        = plotNonlinParams
    plotBlockShift,plotTransformed,nPat,nPatBlock,arrayOfXs,blockSize,\
        nGene,methodLabels\
        = plotBlockShiftParams
    plotGaussAvg,arrayOfXs,nMethod,\
        nPatBlock,nGene,transformName\
        =plotGaussAvgParams
    plotCorr,corrAll,unspikedMean,corrLim,nGcorr,nGene,corrHistBins,nMethod,includeGauss,plotshuffleCorr,methodLabels,arrayOfXs\
        =plotCorrParams
    plotAdjusted,XtestMean,methodLabels,XtestVar,    =plotAdjustedParams
    '''
    Plots 0.0:  Histogram  & caterpillar of transformed exp levels .
    '''
    if plotDists:
        print("Begin plot distributions")
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))
        
        Xbins = np.linspace(minDataVal,maxDataVal,41)
        Xhist = np.histogram(data,bins = Xbins)[0]
        XbinCenters = (Xbins[1:]+Xbins[:-1])/2
        axs[0].plot(XbinCenters, Xhist)
        axs[0].set_title("Dist. of " + transformName + " trans. data")
        axs[0].set_xlabel("(a) " +  transformName+" transfomed exp. level")
        axs[0].set_ylabel("Frequency")
        
        dataMeans = np.mean(data,axis=0)
        Xbins = np.linspace(np.min(dataMeans),np.max(dataMeans),31)
        Xhist = np.histogram(dataMeans,bins = Xbins)[0]
        XbinCenters = (Xbins[1:]+Xbins[:-1])/2
        axs[1].plot(XbinCenters, Xhist)
        axs[1].set_title("Dist. of " + transformName + " trans. data means")
        axs[1].set_xlabel("Mean "+transformName+" transfomed exp. level")
        axs[1].set_ylabel("Frequency")
                
        dataSTDs = np.std(data, axis = 0)
        quintile = (np.arange(nGene)*5)//nGene + 1
        axs[2].scatter(dataMeans, dataSTDs, c=quintile, cmap='viridis', s=10, marker='o')
        cumTmp = np.cumsum(dataSTDs)
        dataSTDavg = (cumTmp[1000:]-cumTmp[:-1000])/1000
        axs[2].plot(dataMeans[500:-500],dataSTDavg,'k--')
        axs[2].set_title("Std. devs. of sorted "+transformName+"  trans. data")
        axs[2].set_xlabel("Mean "+transformName+" transfomed exp. level")
        axs[2].set_ylabel("Standard deviation")
        
        plt.tight_layout()        
        plt.show()

    '''
    Plots 1:  Plot some  representative scale transforms for the nonlinear scale correction. This only needs to be done once, because it doesn't depend on the population divison (but does require LL
    '''
    if plotNonlin:
        print("Begin plot nonlinear scale transforms")
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        pltPats = np.random.choice(nPat, size=nonlinPatNo, replace=False)
        xTmp = np.linspace(minTransformedMean,maxDataVal,30)
        
        for jpat in pltPats:
            # aVec = coefsNL[jpat,:]
            # xFit = xTmp-aVec[0]
            
            # yTmp= 2*xFit / (np.sqrt(aVec[1]**2 + 4*xFit*aVec[2]) + aVec[1])
            yTmp = modelsNL[jpat](xTmp)
            axs[0].plot(xTmp,yTmp)
            # Plot the difference to bring out the difference between different patients            
            axs[1].plot(xTmp,yTmp-xTmp)

        axs[0].set_title("Selected nonlinear scale change functions")
        axs[0].set_xlabel("Expression level")
        axs[0].set_ylabel("Corrected expression level")
        axs[0].plot([4,16],[4,16],'k--',linewidth=3)
        
        
        axs[1].set_title("Selected nonlinear scale change corrections")
        axs[1].set_xlabel("Expression level")
        axs[1].set_ylabel("Correction")

        plt.tight_layout()
        plt.show()
    
    
    ''' Plots 2: per-patient block shifts for different transformed data. Procedure is described in Section 3.2 of first LL paper (see  Overleaf). The only difference is that instead of chopping the data into fixed blocks, we plot a running block average
    This one also only needs to be done once
    '''  
    if plotBlockShift:
        print("Begin plot block shifts")
        pltPats = np.random.choice(nPat, size=nPatBlock, replace=False) # Choose patients to plot
        fig, axs = plt.subplots(4, 2, figsize=(8, 10))

        for jx,Xx in enumerate(arrayOfXs[:4]):
            xTmp = Xx[0]# No spike
            cumTmp = np.cumsum(xTmp,axis=1)
            blockDiffs = (cumTmp[:,blockSize:]-cumTmp[:,:(-blockSize)])/blockSize
            # If not transformed, find the ratio
            if plotTransformed:
                blockDiffs = blockDiffs -np.mean(blockDiffs,axis=0)
            else:
                blockDiffs = blockDiffs/np.mean(blockDiffs,axis=0)
            
            axs[jx,0].plot(np.transpose(blockDiffs[pltPats,::100]))
            axs[jx,0].set_title("Selected sorted block avg. devs., " + methodLabels[jx])
            axs[jx,0].set_xlabel("sorted gene block index")
            axs[jx,0].set_ylabel("Block deviation value")
            
            xTmp = xTmp[:,np.random.permutation(np.arange(nGene))]# No spike
            cumTmp = np.cumsum(xTmp,axis=1)
            blockDiffs = (cumTmp[:,blockSize:]-cumTmp[:,:(-blockSize)])/blockSize
            # Same comment as above
            if plotTransformed:
                blockDiffs = blockDiffs -np.mean(blockDiffs,axis=0)
            else:
                blockDiffs = blockDiffs /np.mean(blockDiffs,axis=0)
            axs[jx,1].plot(np.transpose(blockDiffs[pltPats,::100]))
            axs[jx,1].set_title("Selected unsorted block avg. devs., " + methodLabels[jx])
            axs[jx,1].set_xlabel("unsorted gene block index")
            axs[jx,1].set_ylabel("Block deviation value")
        plt.tight_layout()
        plt.show()
        
        # Plot Gaussian average instead of block average
        # This is smoother, but block avg. is preferred.
        if plotGaussAvg:
            print("Begin plot Gauss averaged shifts")
            # Select nonspiked
            unspikedXs = arrayOfXs[:,0,:,:]
            # Get limits
            meanTmp = np.mean(unspikedXs,axis=1)
            patSelectXs = unspikedXs[:,pltPats,:]
            Xmin,Xmax = np.min(meanTmp),np.max(meanTmp)
            muVals = np.linspace(Xmin,Xmax,21)
            muVals = muVals[1:-1]
            xSigma2 = (1.5*(muVals[1]-muVals[0]))**2    
            
            gDevPat = np.zeros((nMethod,nPatBlock,len(muVals)))
            meanTmp = meanTmp.reshape((nMethod,1,nGene))
              
            for jm, mu in enumerate(muVals):
                gWeights = np.exp(-(meanTmp- mu)**2/(2*xSigma2))
                gWeights = gWeights.reshape((nMethod,1,nGene))
                gAvg = np.sum(gWeights*patSelectXs,axis=2
                              ) / np.sum(gWeights*(1+0*patSelectXs),axis=2)
                gMean = np.sum(gWeights*meanTmp,axis=2
                               ) / np.sum(gWeights*(1+0*meanTmp),axis=2)
                gDevPat[:,:,jm] = gAvg - gMean
     
            for jx,Xx in enumerate(gDevPat):
                
                plt.plot(muVals,np.transpose(Xx))
                plt.title("Gauss averaged deviations,selected patients, " + methodLabels[jx])
                plt.xlabel("Avg. "+transformName+" transformed expression level")
                plt.ylabel("Per-patient deviation value")
                plt.show()

            
    '''
    Plots 3: Correlation distributions.  This corresponds to Fig. 11 in the first paper, but Spearman is used instead of Pearson.  A random sample of nGcorr genes is selected
    This one also only needs to be done once, doesn't depend on population division
    '''
    if plotCorr: # flag for plot
        print("Begin plot correlation distributions")
        arrayOfXsCorr=arrayOfXs[:,0,:,:]
        if corrAll:
            nGcorr = nGene
            ig = range(nGene)
        else:
            ixL = np.logical_and(\
                    unspikedMean>=corrLim[0],
                    unspikedMean<=corrLim[1]).reshape(-1)
            arrayOfXsCorr = arrayOfXsCorr[:,:,ixL]
            nGtmp = arrayOfXsCorr.shape[-1]
            nGcorr = min([nGtmp,nGcorr])
            ig = random.sample(range(nGtmp),nGcorr)
            arrayOfXsCorr = arrayOfXsCorr[:,:,ig]
        corrHist=np.zeros((len(corrHistBins)-1,nMethod))
        corrValsList = []

        if includeGauss:
            arrayOfXsCorr = arrayOfXsCorr[:-1,:,:]
            methodLabelsCorr = methodLabels[:-1]
        else:
            methodLabelsCorr = methodLabels
            
        if plotshuffleCorr:
            # Create an array of random indices for shuffling
            Xnew = arrayOfXsCorr[0].copy()
            nPat = Xnew.shape[0]
            for jc in range(nGcorr):
               rIx  = np.random.permutation(np.arange(nPat))
               Xnew[:,jc] = Xnew[rIx,jc]
            arrayOfXsCorr = np.concatenate((arrayOfXsCorr,Xnew.reshape(1,nPat,-1)))
            methodLabelsCorr = methodLabelsCorr + ["Shuffled"]

            
        for jx,Xx in enumerate(arrayOfXsCorr):
            corrVals = np.ndarray.flatten(spearmanr(Xx).statistic)
            corrVals = corrVals[corrVals < 0.999]
            corrValsList = corrValsList + [corrVals]
            corrHist[:,jx] = np.histogram(corrVals,bins = corrHistBins)[0]
        
        fig, axes = plt.subplots(ncols=1, nrows=2 , figsize=(8,8))
        
        axes[0].plot(corrHistBins[:-1]+1/(len(corrHistBins)-1), corrHist)
        
        if plotshuffleCorr:
            axes[0].set_ylim((0,1.1*np.max(corrHist[:,:-1])))       
        axes[0].set_title("Distribution of Spearman correlations" )
        axes[0].set_xlabel("correlation value",fontsize=10)
        axes[0].set_ylabel("frequency",fontsize=10)
        axes[0].legend(methodLabelsCorr)
        
        nCorrDiffs = len(corrValsList)-1-plotshuffleCorr
        
        corrDiffHist=np.zeros((len(corrHistBins)-1,nCorrDiffs))
        for jc in range(nCorrDiffs):
            corrValsDiff = corrValsList[jc]-corrValsList[nCorrDiffs]
            corrDiffHist[:,jc] =  np.histogram(corrValsDiff,bins = corrHistBins)[0]
        axes[1].plot(corrHistBins[:-1]+1/(len(corrHistBins)-1), corrDiffHist)
        axes[1].set_title("Spearman correlation differences from "+methodLabelsCorr[nCorrDiffs] )
        axes[1].set_xlabel("correlation  difference",fontsize=10)
        axes[1].set_ylabel("frequency",fontsize=10)
        legendText = []
        for leg in methodLabelsCorr[:nCorrDiffs]:
            legendText.append(leg+" - "+methodLabelsCorr[nCorrDiffs])
        axes[1].legend(legendText)
        axes[1].set_xlim((-0.25,0.4))
        plt.tight_layout()
        plt.show()

    if plotAdjusted:
        print("Begin plot adjusted mean and variance")
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        
        axs[0].plot(XtestMean[0,:],XtestMean.T,'.',markersize=0.5)
        axs[0].legend(methodLabels)
        axs[0].set_title("Mean of adjusted data vs. mean "+transformName+" exp. level")
        axs[0].set_xlabel("Mean "+transformName+" trans. expr. level")
        axs[0].set_ylabel("Adjusted mean "+transformName+" trans. expr. level")
    
    
        axs[1].plot(XtestMean[0,:],np.sqrt(XtestVar).T,'.',markersize=0.8)
        axs[1].legend(methodLabels)
        axs[1].set_title("Std. of adjusted data vs. mean "+transformName+" exp. level")
        axs[1].set_xlabel("mean "+transformName+" trans. exp. level")
        axs[1].set_ylabel("standard deviation")
        plt.tight_layout()
        plt.show()
    
        
    XtestVarCum = np.cumsum(XtestVar,axis=1)
    XtestVarBlock = (XtestVarCum[:,1000:] - XtestVarCum[:,:-1000])/1000
    Xvals = (XtestMean[0,1000:]+XtestMean[0,:-1000])/2 
    plt.plot(Xvals,XtestVarBlock.T)

    plt.legend(methodLabels)
    plt.title("Block averaged variance as a function of expression level")
    plt.xlabel("mean log trans. expr. level")
    plt.ylabel("Averaged variance")
    plt.show()
    
    
    # Gaussian
    Xmin,Xmax = np.min(XtestMean),np.max(XtestMean)
    muVals = np.linspace(Xmin,Xmax,22)
    muVals = muVals[1:-1]
    xSigma2 = (1.5*(muVals[1]-muVals[0]))**2    
    
    gVar = np.zeros((nMethod,len(muVals)))
      
    for jm, mu in enumerate(muVals):
        gWeights = np.exp(-(XtestMean- mu)**2/(2*xSigma2))
        gVar[:,jm] = np.sum(gWeights*XtestVar,axis=1
                      ) / np.sum(gWeights*(1+0*XtestVar),axis=1)
 
    plt.plot(muVals,np.transpose(gVar))
    plt.title("Gauss averaged variance "+transformName)
    plt.xlabel("Avg. "+transformName+" transformed expression level")
    plt.ylabel("Gauss averaged variance")
    plt.legend(methodLabels)
    plt.show()

# Not currently used -- needs some debugging
def perDivisionPlots(perDivisionParams):
    nPatDiv,nSpikeLevel,nMethod,arrayOfTstats,methodLabels,\
        spikeLevels,arrayOfXs,transformName,nGene =\
            perDivisionParams
    for jd in range(nPatDiv):
        fig, axes = plt.subplots(ncols=1, nrows=nSpikeLevel , figsize=(4,8))
        # Iterate through each column and plot the empirical density function
        for js in range(nSpikeLevel):
            for jm in range(nMethod):
                col = np.squeeze(arrayOfTstats[jd,jm,js,:])
                sns.distplot(col, bins=30, hist=False, kde=True, ax = axes[js])
            plt.legend(methodLabels)
        plt.suptitle( "t statistic distribution, pop. division #" + str(jd) + ", spike levels  ",str(spikeLevels))
        plt.show()
    
    # Plots showing t statistics versus mean value
    xTmp = arrayOfXs[0,0,:,:]
    xTmp = np.mean(xTmp,axis=0)
    for jd in range(nPatDiv):
        fig, axes = plt.subplots(ncols=1, nrows=nSpikeLevel , figsize=(3,8))
        # Iterate through each column and plot the empirical density function
        for js in range(nSpikeLevel):
            for jm in range(nMethod):
                yTmp = np.squeeze(arrayOfTstats[jd,jm,js,:])
                axes[js].plot(xTmp,yTmp,'.',markersize=0.3)
            plt.legend(methodLabels)
        plt.suptitle( "t statistic vs mean "+transformName+" exp level, pop. division #" + str(jd) + ", spike levels ",str(spikeLevels))
        plt.show()
        
    ## Plot ROC curve for individual.    
    # ROC curves
    # Loop through positive threshold cutoffs
    posLevelVec = np.arange(3.0,2.0,-0.01)
    nPosLevel = len(posLevelVec)
    # This will store all positive detection rates (including false positives)
    arrayOfPosRates = np.zeros((nPatDiv,nMethod,nSpikeLevel,nPosLevel))
    # For each cutoff level, compute the proportion exceeding
    for jl in range(nPosLevel):
        # computes number exceeding
        arrayOfPosRates[:,:,:,jl] = np.sum(arrayOfTstats > posLevelVec[jl], axis = 3)
    # Convert number to proportion
    arrayOfPosRates = arrayOfPosRates/nGene
    
    # Assumes that first spike level is 1
    if nSpikeLevel>1:
        # Plot ROC curves for each patient division.
        for jd in range(nPatDiv):
            fig, axes = plt.subplots(ncols=1, nrows=nSpikeLevel-1 , figsize=(3,8))
            for jm in range(nMethod):
                # False positives for this method (spike level 0)
                xVec = np.squeeze(arrayOfPosRates[jd,jm,0,:])
                for js in range(1,nSpikeLevel):
                    yVec = np.squeeze(arrayOfPosRates[jd,jm,js,:])
                    axes[js-1].plot(xVec,yVec,'.')
                    axes[js-1].set_title("spike level "+str(spikeLevels[js+1]))
                    axes[js-1].set_xlabel("False positive rate")
                    axes[js-1].set_ylabel("True positive rate")
                    
        plt.suptitle( "ROC curves, pop. division #" + str(jd) + ", different spike levels ")
        plt.legend(methodLabels)
        plt.show() 
       
        