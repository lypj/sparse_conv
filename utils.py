""" CDLNet/utils.py
Data manipulation and visualization utilities.
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor
from matplotlib import pyplot as plt

def imgLoad(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def gabor_kernel(a, f0, p=None, x0=None, m=None, cplx=False):
	"""
	generate a batch of gabor filterbank via inverse width (a) and frequency (f0) params
	a  (precision):  (batch, out_chan, in_chan, 2) 
	f0 (center freq): (batch, out_chan, in_chan, 2)
	p  (phase): (batch, out_chan, in_chan)
	h  (output): (batch, out_chan, in_chan, m, m)
	"""
	a = a[:,:,:,None,None,:]
	f0 = f0[:,:,:,None,None,:]

	# phase term
	if p is None:
		p = torch.zeros((a.shape[0],a.shape[1],a.shape[2],1,1), device=a.device)
	else:
		p = p[:,:,:,None,None]

	if m is None:
		m = int(np.ceil(np.maximum(np.max(5*(1/a).numpy()), 1)))
		m = m + 1 - m%2 # ensure odd filter

	if x0 is None:
		x0 = torch.tensor([(m-1)/2,(m-1)/2])[None,None,None,None,None,:]
		x0 = x0.to(a.device)
	
	i = torch.arange(m).to(a.device)
	x = torch.stack(torch.meshgrid(i,i, indexing='ij'), dim=2)[None,None,...]
	if cplx:
		h = torch.exp( -torch.sum((a*(x-x0))**2, dim=-1) ) * \
			torch.exp(1j*2*np.pi*(torch.sum(f0*(x-x0), dim=-1) + p))
		return h

	h = torch.exp( -torch.sum((a*(x-x0))**2, dim=-1) ) * \
	    torch.cos(2*np.pi*(torch.sum(f0*(x-x0), dim=-1) + p))
	return h

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled 
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255), sigma

def pre_process(x, stride):
	""" image preprocessing: stride-padding and mean subtraction.
	"""
	params = []
	# mean-subtract
	xmean = x.mean(dim=(2,3), keepdim=True)
	x = x - xmean
	params.append(xmean)
	# pad signal for stride
	pad = calcPad2D(*x.shape[2:], stride)
	x = F.pad(x, pad, mode='reflect')
	params.append(pad)
	return x, params

def post_process(x, params):
	""" undoes image pre-processing given params
	"""
	# unpad
	pad = params.pop()
	x = unpad(x, pad)
	# add mean
	xmean = params.pop()
	x = x + xmean
	return x

def calcPad1D(L, M):
	""" Return pad sizes for length L 1D signal to be divided by M
	"""
	if L%M == 0:
		Lpad = [0,0]
	else:
		Lprime = np.ceil(L/M) * M
		Ldiff  = Lprime - L
		Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
	return Lpad

def calcPad2D(H, W, M):
	""" Return pad sizes for image (H,W) to be divided by size M
	(H,W): input height, width
	output: (padding_left, padding_right, padding_top, padding_bottom)
	"""
	return (*calcPad1D(W,M), *calcPad1D(H,M))

def conv_pad(x, ks, mode):
	""" Pad a signal for same-sized convolution
	"""
	pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
	return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
	""" Remove padding from 2D signalstack"""
	if pad[3] == 0 and pad[1] > 0:
		return I[..., pad[2]:, pad[0]:-pad[1]]
	elif pad[3] > 0 and pad[1] == 0:
		return I[..., pad[2]:-pad[3], pad[0]:]
	elif pad[3] == 0 and pad[1] == 0:
		return I[..., pad[2]:, pad[0]:]
	else:
		return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

def visplot(images,
	        grid_shape=None,
	        crange = (None,None),
	        primary_axis = 0,
	        titles	 = None,
	        colorbar = False,
	        cmap = 'gray'):
	""" Visual Subplot, adapted from Amir's code.
	Plots array of images in grid with shared axes.
	Very nice for zooming.
	"""
	if grid_shape is None:
		grid_shape = (1,len(images))
	fig, axs = plt.subplots(*grid_shape,sharex='all',sharey='all',squeeze=False)
	nrows, ncols = grid_shape
	# fill grid row-wise or column-wise
	if primary_axis == 1:
		indfun = lambda i,j: j*nrows + i
	else:
		indfun = lambda i,j: i*ncols + j
	im_list = []
	for ii in range(nrows):
		for jj in range(ncols):
			ind = indfun(ii,jj)
			if ind < len(images):
				if type(images[ind])==torch.Tensor:
					img = images[ind].detach().permute(1,2,0).squeeze()
				else:
					img = images[ind].squeeze()
				im = axs[ii,jj].imshow(img,
				                       cmap   = cmap,
				                       aspect = 'equal',
				                       interpolation = None,
				                       vmin = crange[0],
				                       vmax = crange[1])
				if colorbar:
					fig.colorbar(im,
					             ax       = axs[ii,jj],
					             fraction = 0.046,
					             pad      = 0.04)
			axs[ii,jj].axis('off')
			if (titles is not None) and (ind < len(titles)):
				axs[ii,jj].set_title(titles[ind])
	return fig
