__author__ = 'Maulik K. Nariya'
__copyright__ = 'CC-BY-4.0'
__date__ = 'July 2023'

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import anndata
import gseapy as gp
import utils

def pca_genexp(adata, genes, shift=0.83):
	'''
	Preforms PCA of gene expression data
	'''
    dft = pd.DataFrame(index=adata.obs['cell_cycle_theta'].index)
    dft['theta'] = [(t - shift) % 1 for t in adata.obs['cell_cycle_theta'].tolist()]
    dfs = adata.to_df(layer='Ms').T[adata.to_df(layer='Ms').T.index.isin(genes)].T
    pca = PCA(n_components=2)
    pca.fit(dfs.values)
    Xpca = pca.transform(dfs.values)
    return Xpca, dft
    
def res_var(adata, gene, shift):
	'''
	Calculates the moving averages, uker and sker, and residual variances, uvar and svar
	'''
	u, s, t = utils.load_unsp(adata, gene, shift=shift)
	tsm, usm = utils.kernel_smooth_periodic(x=t, y=u, fwhm=0.5, num_pts=len(t))
	tsm, ssm = utils.kernel_smooth_periodic(x=t, y=s, fwhm=0.5, num_pts=len(t))
	uker = np.asarray([usm[(abs(t[i] - tsm).argmin())] for i in range(len(t))])
	sker = np.asarray([ssm[(abs(t[i] - tsm).argmin())] for i in range(len(t))])
	uvar = np.square(u - uker)
	svar = np.square(s - sker)
	return uker, uvar, sker, svar
	
def time_maxrates(df):
	'''
	Returns the timing of max rate for the genes 
	'''
	df.columns = [float(x) for x in df.columns]
	tmaxr = []
	for gn in df.index.tolist():
	    tmaxr.append(2*np.pi*float(df.columns[df.loc[gn].tolist().index(df.loc[gn].max())]))
	return np.asarray(tmaxr)

def genes_tmaxr(dfnorm):
	'''
	Sorts the genes based on the timing of max rates
	'''
	rmax_indc = []
	genes = dfnorm.index.tolist()
	for r in range(dfnorm.shape[0]):
	    rmax_indc.append(np.where(np.asarray(dfnorm.iloc[r])==1)[0][0])
	rmax_indc, rmax_genes = (list(x) for x in zip(*sorted(zip(rmax_indc, genes))))
	rmax_indc = np.asarray(rmax_indc)
	rmax_genes = np.asarray(rmax_genes)
	return rmax_genes
	
def gene_waves(dfnorm, token=None, outdir='../results/'):
	'''
	Returns lists of genes, w1, w2, w3, such that w1 consists of genes that achieve max rates
	during G1, w2 genes that achieve max rates during S, and w3 genes that achieve max rates
	during G2/M
	'''
	rmax_indc = []
	genes = dfnorm.index.tolist()
	for r in range(dfnorm.shape[0]):
	    rmax_indc.append(np.where(np.asarray(dfnorm.iloc[r])==1)[0][0])
	rmax_indc, rmax_genes = (list(x) for x in zip(*sorted(zip(rmax_indc, genes))))
	rmax_indc = np.asarray(rmax_indc)
	rmax_genes = np.asarray(rmax_genes)
	w1 = rmax_genes[np.where(rmax_indc<=25)[0]]
	w2 = rmax_genes[np.where((rmax_indc>25) & (rmax_indc<=63))[0]]
	w3 = rmax_genes[np.where(rmax_indc>63)[0]]
	if token is not None:
		for i, w in zip([1, 2, 3], [w1, w2, w3]):
			with open('%s%s%s.txt'%(outdir, token, i), 'w') as outFile:
				for x in w: outFile.write('%s\n'%x)
			outFile.close()
	return w1, w2, w3
	
def gsea(file_names, go_term='GO_Biological_Process_2018', organism='mouse',
		min_set_size=20):
	'''
	Performs gene set enrichment analysis on "waves" of genes, list of genes should be stored in
	text files
	'''
	names = ["W1", "W2", "W3"]
	drop_cols = ["Gene_set", "P-value", "Old P-value", "Old Adjusted P-value", 
		         "Odds Ratio", "Combined Score", "Genes"]
	keep_cols = ["Overlap", "Adjusted P-value", "ratio", "set_size", "name"]
	df = []
	for fl, n in zip(file_names, names):
		enr = gp.enrichr(gene_list=fl,
		                 gene_sets=[go_term],
		                 organism=organism, 
		                 outdir=None)
		                 
		enr.results["ratio"] = [float(x.split("/")[0])/float(x.split("/")[1]) for x in enr.results["Overlap"]]
		enr.results["set_size"] = [float(x.split("/")[1]) for x in enr.results["Overlap"]]
		enr.results["name"] = [n]*enr.results.shape[0]
		df.append(enr.results[(enr.results.ratio>0.1) & 
		            (enr.results["Adjusted P-value"]<0.05) & 
		            (enr.results["set_size"]>min_set_size)].sort_values(by="set_size", ascending=False).sort_values(by="ratio", ascending=False))

	dfen = pd.concat([df[0], df[1],df[2]], ignore_index=True)
	return dfen

def bicprobab(dfloss, n):
	'''
	Caluclates the porbabilities, based on the Bayesian inference criterion, of selecting
	alternate models
	'''
	cols = dfloss.columns.tolist()
	dfbic = pd.DataFrame()
	dfbic_prime = pd.DataFrame()
	dfbic[cols[0]] = 23 * np.log(n) + 2 * dfloss[cols[0]]
	dfbic[cols[1]] = 13 * np.log(n) + 2 * dfloss[cols[1]]
	dfbic[cols[2]] = 13 * np.log(n) + 2 * dfloss[cols[2]]
	dfbic[cols[3]] = 3 * np.log(n) + 2 * dfloss[cols[3]]
	dfbic_prime = dfbic.subtract(dfbic.min(axis=1), axis=0)
	dfprobab = np.exp(-dfbic_prime)
	dfprobab  = dfprobab.div(dfprobab.sum(axis=1), axis=0)
	dfplot = pd.DataFrame(index=dfprobab.mean().index, columns=['probability'])
	dfplot['probability'] = dfprobab.mean().tolist()
	return dfplot

