""" CDLNet/net.py
Definition of CDLNet module.
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import conv, solvers, utils, nle

class CDLNet(nn.Module):
	""" Convolutional Dictionary Learning Network:
	Interpretable denoising DNN with adaptive thresholds for robustness.
	"""
	def __init__(self,
	             num_filters = 64,   # num. filters in each filter bank operation
	             num_inchans = 1,
	             filter_size = 7,    # square filter side length
	             stride = 1,         # strided convolutions
	             iters  = 3,         # num. unrollings
	             tau0   = 1e-2,      # initial threshold
	             adaptive = False,   # noise-adaptive thresholds
	             gabor_init = False, # initialize weights with gabor filters
	             init = True):       # False -> use power-method for weight init
		super(CDLNet, self).__init__()
		
		# -- OPERATOR INIT --
		if gabor_init:
			a     = torch.randn(1,num_filters,num_inchans,2)
			f0    = torch.randn(1,num_filters,num_inchans,2)
			psi   = torch.randn(1,num_filters,num_inchans)
			alpha = torch.randn(1,num_filters,num_inchans,1,1)
			W = (alpha*utils.gabor_kernel(a, f0, p=psi, m=filter_size)).sum(dim=0)
		else:
			W = torch.randn(num_filters,num_inchans,filter_size,filter_size)

		def conv_gen():
			C = conv.Conv2d(num_inchans, num_filters, filter_size, stride)
			C.weight = W.clone()
			return C
		def convT_gen():
			if stride == 1:
				C = conv.Conv2d(num_filters, num_inchans, filter_size, stride)
				C.weight = conv.adjoint(W).clone()
			else:
				C = conv.ConvAdjoint2d(num_filters, num_inchans, filter_size, stride)
				C.weight = W.clone()
			return C
		self.D = convT_gen()
		self.A = nn.ModuleList([conv_gen()  for _ in range(iters)])
		self.B = nn.ModuleList([convT_gen() for _ in range(iters)])

		# Don't bother running code if initializing trained model from state-dict
		if init:
			with torch.no_grad():
				print("Running power-method on initial dictionary...")
				L, _, _ = solvers.powerMethod(lambda x: self.B[0](self.A[0](x)),
											  torch.rand(1,num_inchans,128,128),
											  num_iter= 100,
											  verbose = False)
				print(f"Done. L={L:.3e}.")
				if L < 0:
					print("STOP: something is very very wrong...")
					sys.exit()
		else:
			L = 1

		# spectral normalization
		self.D.weight = self.D.weight / np.sqrt(L)
		for ii in range(iters):
			self.A[ii].weight = self.A[ii].weight / np.sqrt(L)
			self.B[ii].weight = self.B[ii].weight / np.sqrt(L)
		
		# learned thresholds
		self.tau = nn.ParameterList([nn.Parameter(tau0*torch.ones(2,num_filters,1,1)) for _ in range(iters)])

		# set parameters
		self.iters = iters
		self.tau0  = tau0
		self.adaptive  = adaptive
		self.stride    = stride
		self.num_filters = num_filters
		self.filter_size = filter_size

	@torch.no_grad()
	def project(self):
		""" \ell_2 ball projection for filters, R_+ projection for thresholds
		"""
		W = self.D.weight; norm2W = torch.norm(W,dim=(2,3),keepdim=True)
		self.D.weight = W * torch.clamp(1/norm2W, max=1)
		for ii in range(self.iters):
			W = self.A[ii].weight; norm2W = torch.norm(W,dim=(2,3),keepdim=True)
			self.A[ii].weight = W * torch.clamp(1/norm2W, max=1)
			W = self.B[ii].weight; norm2W = torch.norm(W,dim=(2,3),keepdim=True)
			self.B[ii].weight = W * torch.clamp(1/norm2W, max=1)
			self.tau[ii].clamp_(0.0) 

	def forward(self, y, sigma_n=None):
		""" LISTA + D w/ noise-adaptive thresholds
		""" 
		yp, params = utils.pre_process(y, self.stride)
		# THRESHOLD SCALE-FACTOR c
		c = 0 if sigma_n is None or not self.adaptive else sigma_n/255.0
		# LISTA
		z = ST(self.A[0](yp), self.tau[0][:1] + c*self.tau[0][1:2])
		for k in range(1, self.iters):
			z = ST(z - self.A[k](self.B[k](z) - yp), self.tau[k][:1] + c*self.tau[k][1:2])
		# DICTIONARY SYNTHESIS
		xphat = self.D(z)
		xhat  = utils.post_process(xphat, params)
		return xhat, z

def ST(x,T):
	""" Soft-thresholding operation. 
	Input x, threshold T.
	"""
	return x.sign()*F.relu(x.abs()-T)

