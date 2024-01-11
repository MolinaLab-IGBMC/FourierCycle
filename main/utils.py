__author__ = 'Maulik K. Nariya'
__date__ = 'March 2023'

import numpy as np
import pandas as pd
import anndata
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

def kernel_smooth_periodic(x, y, fwhm=None, num_pts=None):
    '''
    Performs kernel smoothing using a periodic Gaussian (von Mises) kernel
    Parameters:
    -----------
    x : 1D array, dtype=float
    	x-values
    y : 1D array, dtype=float
    	y-values
    fwhm : float 
    	full width at half maximum (bandwidth)
         if None, then will be equal to (xmax-xmin) / 100
    Returns:
    --------
    xsm, ysm : float
    	kernel smoothed values of x and y  
    '''
    x, y = zip(*sorted(zip(x, y)))
    x = np.asarray(x)
    y = np.asarray(y)
    if fwhm is None:
        fwhm = 2.0*np.pi*0.1*(max(x) - min(x))
    if num_pts is None:
        num_pts = int((max(x) - min(x)) / fwhm)
    delx = (max(x) - min(x)) / num_pts          
    
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    xsm = np.linspace(min(x), max(x), num_pts)
    sigma = fwhm2sigma(fwhm)
    dist = -2.0*np.pi*cdist(xsm.reshape(len(xsm), 1), x.reshape(len(x), 1))
    kernel = np.exp(np.cos(dist)/(sigma**2))
    norm_kernel = normalize(kernel, axis=1, norm='l1')
    ysm = np.dot(norm_kernel, y)
    return xsm, ysm


def load_unsp(adata, gene, phase=True, shift=None):
    '''
    Loads Anndata object, returns unspliced and spliced moments and the 
    estimated cell cycle phases theta (theta belongs to (0,1)) for a given gene
    Parameters:
	----------
	adata: Anndata object
		must contain "n_counts" and "cell_cycle_theta" in adata.obs and
		Mu and Ms in adata.layers
	gene: str
		gene name
	phase: bool
		if true returns the cell cycle phase along with unspliced and spliced reads
	shift: float
		if not None the the cell cycle phases are shifted by this value:
	Returns:
	--------
	u: float
		unspliced
	s: float 
		spliced
	t: float
		theta
    '''
    gnidx = np.where(adata.var['Accession'].index==gene)[0][0]
    sp = adata.layers['Ms'][:,gnidx]
    un = adata.layers['Mu'][:,gnidx]
    if type(sp) is not np.ndarray:
        sp = sp.toarray().flatten()
    if type(un) is not np.ndarray:
        un = un.toarray().flatten()
    if phase is True:
        th = adata.obs["cell_cycle_theta"]
        if shift is not None:
        	th = [(t - shift) % 1 for t in th]
        return np.asarray(un), np.asarray(sp), np.asarray(th)
    else:
        return np.asarray(un), np.asarray(sp)
        

def varmu_slope(adata, gene):
	'''
	Caclulates the slopes between the residual variance and the moving averages of 
	spliced and unspliced for a gene
	Parameters:
	-----------
	adata: Anndata object
		must contain "n_counts" and "cell_cycle_theta" in adata.obs and
		Mu and Ms in adata.layers
	gene: str
		gene name
	Returns:
	--------
	uslope: float
		slope for unspliced
	sslope: float 
		slope for spliced
	'''	
	dfu = adata.to_df(layer='Mu')
	dfs = adata.to_df(layer='Ms')
	fu = dfu.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	fs = dfs.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	# Rescale u and s such that sum over genes is equal to n_counts for every cell, 
	# preserve the fractional contributions from u and s
	dfusc = fu * dfu.div(dfu.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfssc = fs * dfs.div(dfs.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfgus = pd.concat([adata.obs['cell_cycle_theta'], dfusc.T.loc[gene], dfssc.T.loc[gene]], axis=1)
	dfgus.columns = ['theta', 'unspliced', 'spliced']
	th = dfgus['theta'].tolist()
	tsm, usm = kernel_smooth_periodic(x=th, y=dfgus['unspliced'], fwhm=0.5, num_pts=dfgus.shape[0])
	tsm, ssm = kernel_smooth_periodic(x=th, y=dfgus['spliced'], fwhm=0.5, num_pts=dfgus.shape[0])
	uker = [usm[(abs(th[i] - tsm).argmin())] for i in range(len(th))]
	sker = [ssm[(abs(th[i] - tsm).argmin())] for i in range(len(th))]
	uvar = np.square(dfgus['unspliced'] - uker)
	svar = np.square(dfgus['spliced'] - sker)
	if np.sum(uker)==0.0:
		uslope = 0.0
	else:
		uslope = np.sum(uvar) / np.sum(uker)
	if np.sum(sker)==0.0:
		sslope = 0.0
	else:	
		sslope = np.sum(svar) / np.sum(sker)
	return uslope, sslope
	

def totmRNA(adata_file, slopes_file, shift=None, print_fracs=False):
	'''
	Calibrates the adata, returns a data frame with theta and totmRNA
	
	Parameters:
	-----------
	adata_file: str
		Anndata file
	slopes_file: str
		file with the slope parameter for every gene 
	Returns:
	--------
	dftot: pandas DataFrame
		columns= [theta", "totmRNA"]
	'''
	adata = anndata.read_h5ad(adata_file)
	dfsp = pd.read_csv(slopes_file, index_col=0)
	dfu = adata.to_df(layer='Mu')
	dfs = adata.to_df(layer='Ms')
	fu = dfu.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	fs = dfs.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	dfusc = fu * dfu.div(dfu.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfssc = fs * dfs.div(dfs.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfucal = dfusc.T.sort_index().T.div(dfsp['u-slope'], axis=1)
	dfscal = dfssc.T.sort_index().T.div(dfsp['s-slope'], axis=1)
	if print_fracs is True:
		fucal = dfucal.sum(axis=1) / (dfucal.sum(axis=1) + dfscal.sum(axis=1))  
		fscal = dfscal.sum(axis=1) / (dfucal.sum(axis=1) + dfscal.sum(axis=1))
		print ('Fraction unspliced = %.2f ± %.2f'%(np.mean(fucal), np.std(fucal)))
		print ('Fraction spliced = %.2f ± %.2f'%(np.mean(fscal), np.std(fscal)))
	dftot = pd.concat([adata.obs['cell_cycle_theta'], dfucal.sum(axis=1) + dfscal.sum(axis=1)], axis=1) 
	dftot.columns = ['theta', 'totmRNA'] 
	if shift is not None:
		dftot['theta'] = [(t - shift) % 1 for t in dftot['theta']]
	return dftot, dfucal, dfscal
	

def load_unsp_cal(adata, gene, slopes_file, shift=None, norm=False):
	'''
    Loads Anndata object, returns calibrated unspliced and spliced moments and the 
    estimated cell cycle phases theta (theta belongs to (0,1)) for a given gene
    Parameters:
	-----------
	adata: Anndata object
		must contain "n_counts" and "cell_cycle_theta" in adata.obs and
		Mu and Ms in adata.layers
	gene: str
		gene name
	slopes_file: str
		file with the slope parameter for every gene 
	shift: float
		if not None the the cell cycle phases are shifted by this value
	norm: bool
		if True, the normalizes the calibrated counts such that all cells have 
		same unspliced and spliced molecules in total
	Returns:
	--------
	u: float
		unspliced
	s: float 
		spliced
	t: float
		theta
    '''
	dfsp = pd.read_csv(slopes_file, index_col=0)
	dfu = adata.to_df(layer='Mu')
	dfs = adata.to_df(layer='Ms')
	fu = dfu.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	fs = dfs.sum(axis=1).div((dfu.sum(axis=1) + dfs.sum(axis=1))).mean()
	dfusc = fu * dfu.div(dfu.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfssc = fs * dfs.div(dfs.sum(axis=1), axis=0).multiply(adata.obs['n_counts'], axis=0)
	dfucal = dfusc.T.sort_index().T.div(dfsp['u-slope'], axis=1)
	dfscal = dfssc.T.sort_index().T.div(dfsp['s-slope'], axis=1)
	dfucal = dfucal.fillna(0.0)
	dfscal = dfscal.fillna(0.0)
	dftot = dfucal + dfscal
	if norm is True:
		dfucal = dfucal.div(dftot.sum(axis=1), axis=0)*np.mean(dftot.sum(axis=1))
		dfscal = dfscal.div(dftot.sum(axis=1), axis=0)*np.mean(dftot.sum(axis=1)) 
	dfcal = pd.concat([adata.obs['cell_cycle_theta'], dfucal[gene], dfscal[gene]], axis=1)
	dfcal.columns = ['theta', 'unspliced', 'spliced']
	if shift is not None:
		dfcal['theta'] = [(t - shift) % 1 for t in dfcal['theta']]
	u = np.asarray(dfcal['unspliced'].tolist())
	s = np.asarray(dfcal['spliced'].tolist())
	t = np.asarray(dfcal['theta'].tolist())
	return u, s, t
