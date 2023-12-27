__author__ = 'Maulik K. Nariya'
__copyright__ = 'MIT license'
__date__ = 'March 2023'

import numpy as np
from scipy import optimize
import warnings

class FourierSolve():
	'''
	Uses the Fourier series approach to solve dynamical system 
	of splcied-unspliced abundances.
	du(t)/dt = alpha(t) - beta * u(t)
	ds(t)/dt = beta * u(t) - gamma(t) * s(t) 
	alpha : production rate, 
	beta : splicing rate, 
	gamma : degradation rate
	Parameters:
	-----------
	u : float, array-like, shape = n_cells
		Unspliced reads.
	s : float, array-like, shape = n_cells
		Spliced reads.
	t : float, array-like, shape = n_cells
		Pseuduotime or the transcriptional phase of the cells.
	Nt : int, default = 5
		The Fourier series truncates at this value.
	n_rs : int, default = 100
		size of the resampled t array
		t_rs = np.linspace(max(t), min(t), n_rs)
		alpha, gamma are evaluated in this resampled basis
	l1 : float, default = 1
		penalty on the residuals on spliced and unspliced fits
	l2 : float, default = 1
		penalty on negative rates alpha and gamma
	l3 : float, default = 0.01
		penalty of the pathlength on alpha and gamma
	tol : float, default = 1e-8
		tolerance for the error function 
	itol : float, default = 1e-4
		tolerance for the imaginary parts, 
		issues a warning if the sum of imaginary parts is greater than this   
	'''
	
	def __init__(self, u, s, t, Nt=5, n_rs=100, l1=1, l2=1, l3=0.01, tol=None, itol=1e-4, method='BFGS'):
		self.u = np.asarray(u)
		self.s = np.asarray(s)
		self.t = np.array(t)
		self.Nt = Nt
		self.n_rs = n_rs
		self.t_rs = np.linspace(min(self.t), max(self.t), self.n_rs)
		self.l1 = l1
		self.l2 = l2
		self.l3 = l3
		self.tol = tol
		self.itol = itol
		self.method = method
		self.E = self.Ematrix(self.t)
		self.E_rs = self.Ematrix(self.t_rs)
		self.R = self.Rmatrix()
		self.C = self.Cmatrix()
	
	
	def Ematrix(self, t):
		'''
		Generates the E matrix, e^{i*Nt*theta}, for given list of transcriptional phases
		Parameters:
		-----------
		t : float, array-like, shape = n_cells
			Pseuduotime or the transcriptional phase of the cells.
		Returns:
		--------
		E : complex, array-like, shape = (2Nt+1, n_cells)
		'''
		n = [x - self.Nt for x in range(2 * self.Nt + 1)]
		E = np.ndarray(shape=(len(t), 2*self.Nt+1), dtype=complex)
		for i in range(E.shape[0]):
			for j in range(E.shape[1]):
				E[i,j] = np.exp(complex(0, 2 * n[j] * t[i] * np.pi))
		return E
	
	
	def Gmatrix(self, gamma):
		'''
		Generates the G matrix for the given complex Fourier coefficents of gamma
		Parameters:
		-----------
		gamma: complex, array-like, shape=2*Nt+1
		
		Returns:
		-------
		G: complex, array-like, shape = 
		'''
		n = [x - self.Nt for x in range(2 * self.Nt + 1)]
		G = []
		for i in range(len(n)):#loop over n
			row = []
			for j in range(len(n)): #loop over k
				if n[i]==n[j]:
					x = complex(0, n[i])
				else:
					x = complex(0, 0)
				if n[i] - n[j] >= -self.Nt and n[i] - n[j] <= self.Nt:
					idx = n.index(n[i] - n[j])
					y = gamma[idx]
				else:
					y = complex(0, 0)
				row.append(x + y)
			G.append(row)
		return np.asarray(G)
	
	
	def Rmatrix(self):
		'''
		Return a matrix transforms complex Fourier coefficients to real
		aR = np.matmul(R, rates[0:2*Nt+1])
		gR = np.matmul(R, rates[2*Nt+1:4*Nt+2])
		'''
		Nt = self.Nt
		x = np.asarray([1] + [0]*2*Nt)
		R = []
		for i in range(2*Nt+1):
			if i<Nt:
				R.append(list(np.roll(x, i+1)))
			elif i>Nt:
				R.append(list(np.roll(x, i)))
			else:
				R.append(list(x))
		R = np.asarray(R)
		return R
	
	
	def Cmatrix(self):
		'''
		Returns a matrix that transforms real Fourier coefficients to complex
		aC = np.matmul(C, aR)
		gC = np.matmul(C, gR)
		'''
		Nt = self.Nt
		x = [complex(0, 0)] * (Nt-1) + [complex(0.5, 0)] + [complex(0, 0)] + [complex(0, 0)] * (Nt-1) + [complex(0, 0.5)]
		z = [complex(0, 0)] * Nt + [complex(1, 0)] + [complex(0, 0)] * Nt
		C = []
		for i in range(2*Nt+1):
			if i<Nt:
				C.append(np.roll(x, -i))
			elif i>Nt:
				C.append(np.roll(np.conjugate(x), i+1))
			else:
				C.append(z)
		C = np.asarray(C)
		return C
	
	
	def gene_model(self, rates):
		'''
		Function that approximates the unpliced and spliced abundances as a 
		Fourier series sum
		Parameters:
		-----------
		rates : float, array-like, shape = 4*Nt+3
			Fourier coefficients of the rate parameters of the dynamical model
			rates = [alpha0, alphaA1, alphaA2, ... alphaANt, alphaB1, alphaB2, ... alphaBNt, 
					gamma0, gammaA1, gammaA2, ... gammaANt, gammaB1, gammaB2, ... gammaBNt,
					beta]
		Returns:
		-------
		un_mod: float, array-like, shape=n_cells
			Fourier series sum approximation for unspliced abundances 
		sn_mod: float, array-like, shape=n_cells
			Fourier series sum approximation for spliced abundances
		'''
		Nt = self.Nt
		n_rs = self.n_rs
		rates = np.asarray(rates)
		n = [x - Nt for x in range(2 * Nt + 1)]
		b = rates[-1]
		aR = np.matmul(self.R, rates[0:2*Nt+1])		# real coefficients of alpha
		gR = np.matmul(self.R, rates[2*Nt+1:4*Nt+2])# real coefficients of alpha
		aC = np.matmul(self.C, aR)                  # complex coefficients of alpha
		gC = np.matmul(self.C, gR)                  # complex coefficients of gamma
		daC_dt = np.array([complex(0, x) * y for x, y in zip(n, aC)]) # first derivavtive of alpha w.r.t.theta
		dgC_dt = np.array([complex(0, x) * y for x, y in zip(n, gC)]) # first derivavtive of gamma w.r.t.theta
		un = np.array([y / (complex(b, x)) for x, y in zip(n, aC)])  # un = alpha/(beta+i*n)
		G = self.Gmatrix(gC)                                         
		sn = b * np.matmul(np.linalg.inv(G), un) 
		u_mod = np.matmul(self.E, un)
		s_mod = np.matmul(self.E, sn)
		a_mod = np.matmul(self.E_rs, aC)
		g_mod = np.matmul(self.E_rs, gC)
		da_dt_mod = np.matmul(self.E_rs, daC_dt)
		dg_dt_mod = np.matmul(self.E_rs, dgC_dt)
		
		if abs(sum(u_mod.imag)) > self.itol:
			warnings.warn("Sum of the imaginary part of 'unspliced' solution is = %s"%sum(u_mod.imag), UserWarning)
		if abs(sum(s_mod.imag)) > self.itol: 
			warnings.warn("Sum of the imaginary part of 'spliced' solution is = %s"%sum(s_mod.imag), UserWarning)
		if abs(sum(a_mod.imag)) > self.itol:
			warnings.warn("Sum of the imaginary part of 'alpha' solution is = %s"%sum(a_mod.imag), UserWarning)
		if abs(sum(g_mod.imag)) > self.itol: 
			warnings.warn("Sum of the imaginary part of 'gamma' solution is = %s"%sum(g_mod.imag), UserWarning) 
		
		return u_mod.real, s_mod.real, a_mod.real, g_mod.real, da_dt_mod.real, dg_dt_mod.real, b
	
	   
	def loss_func(self, rates):
		'''
		Calculates the residual between the Fourier series approximation and the data
		for unspliced and spliced, penalizes negative rates
		Parameters:
		-----------
		rates : float, array-like, shape = 2*Nt+2
			Fourier coefficients of the rate parameters of the dynamical model
			rates = [alpha0, alphaA1, alphaA2, ..., alphaANt, alphaB1, alphaB2, ... alphaBNt,
			gamma0, gammaA1, gammaA2, ..., gammaANt, gammaB1, gammaB2, ... gammaBNt,
			beta]
		Returns:
		--------
		error : float, array-like, shape=len(self.t)
			difference between the unspliced and spliced and their corresponding Fourier
			approximations obtained using the function gene_model()
		'''
		n_rs = self.n_rs
		t_rs = self.t_rs
		l1 = self.l1
		l2 = self.l2
		l3 = self.l3
		
		u_mod, s_mod, a_mod, g_mod, da_dt_mod, dg_dt_mod,b_mod = self.gene_model(rates)
		
		# Error function to be minimized
		# Residuals:  (u_data - u_model)**2 + (s_data - s_model)**2
		u_res = l1 * len(self.u) * np.log(np.sum(np.square((u_mod - self.u))))
		s_res = l1 * len(self.s) * np.log(np.sum(np.square((s_mod - self.s))))
		
		# Penalize negative rates: (alpha_model)**2 * H(alpha_model) +  
		#(gamma_model)**2 * H(gamma_model)
		a_neg = l2 * np.sum(np.square(a_mod[a_mod<0]))
		g_neg = l2 * np.sum(np.square(g_mod[g_mod<0]))
		
		# Penalize the pathlength of the rates alpha and gamma  
		delt = (max(self.t) - min(self.t)) / (self.n_rs)
		a_len = l3 * delt * np.sum(np.sqrt((np.ones(shape=n_rs) + da_dt_mod ** 2)))
		g_len = l3 * delt * np.sum(np.sqrt((np.ones(shape=n_rs) + dg_dt_mod ** 2)))
		
		error = u_res + s_res + a_neg + g_neg + a_len + g_len
		
		return  error

				
	def fit_rates(self, syn_off=False, deg_off=False):
		'''
		Performs the least squares optimization for the rates
		Parameters:
		-----------
		syn_off : bool, default False
			if True then the Fourier coefficients for synthesis are set to zero
		deg_off : bool, default False
			if True then the Fourier coefficients for degradation are set to zero
		Note: if either syn_off or deg_off is True then the variable "methods" should be set to 
		solver compatiable with bounds, e.g. "L-BFGS-B"
		Returns:
		--------
		rates : float, array-like, shape = 4*Nt+3
			fitted rates
		'''
		rates0 = np.zeros(shape=4*self.Nt+3)
		rates0[-1] = (2.0 * np.pi * 5.0)/(14.0 * 60.0)
		if np.mean(self.u) < 1e-4:
			umean = 1e-4
		else:
			umean = np.mean(self.u)
		if np.mean(self.s) < 1e-4:
			smean = 1e-4
		else:
			smean = np.mean(self.s) 
		rates0[0] = rates0[-1] * umean
		rates0[2*self.Nt+1] = rates0[0] / smean
		
		if syn_off is False and deg_off is False:
			bounds = None
		elif syn_off is True and deg_off is False:
			bounds = [(-np.inf, np.inf)]
			for i in range(2*self.Nt):
				bounds.append((0.0, 0.0))
			bounds.append((-np.inf, np.inf))
			for i in range(2*self.Nt):
				bounds.append((-np.inf, np.inf))
			bounds.append((-np.inf, np.inf))
			bounds = tuple(bounds)
		elif syn_off is False and deg_off is True:
			bounds = [(-np.inf, np.inf)]
			for i in range(2*self.Nt):
				bounds.append((-np.inf, np.inf))
			bounds.append((-np.inf, np.inf))
			for i in range(2*self.Nt):
				bounds.append((0.0, 0.0))
			bounds.append((-np.inf, np.inf))
			bounds = tuple(bounds)
		elif syn_off is True and deg_off is True:
			bounds = [(-np.inf, np.inf)]
			for i in range(2*self.Nt):
				bounds.append((0.0, 0.0))
			bounds.append((-np.inf, np.inf))
			for i in range(2*self.Nt):
				bounds.append((0.0, 0.0))
			bounds.append((-np.inf, np.inf))
			bounds = tuple(bounds)
		res = optimize.minimize(fun=self.loss_func, x0=rates0, bounds=bounds, tol=self.tol, method=self.method)
		return res
