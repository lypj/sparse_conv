#!/usr/bin/env python3
""" CDLNet/analyze.py
Executuable for analyzing trained CDLNet models.
Usage: $ ./analyze.py /path/to/args.json [--options] 
see $ ./analyze.py --help for list of options.
Note: blind testing requires option --blind=\"PCA\" or --blind=\"MAD\"
"""
import os, sys, json, copy, time
from pprint import pprint
import numpy as np
from numpy.fft import fftshift, fft2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

import utils, data, nle, train
from utils import visplot

def main(args, img="Set12/09.png", noise_std=25, loader=None, dictionary=True, passthrough=True, blind=False, show=False, save=False, freq=True):
	ngpu = torch.cuda.device_count()
	device = torch.device("cuda:0" if ngpu > 0 else "cpu")
	print(f"Using device {device}.")
	model_args, paths = [args[item] for item in ['model','paths']]
	model, _, _, epoch0 = train.initModel(args, device=device)
	model.eval()
	# ---------------------------------------------------
	# ---------------------------------------------------
	with torch.no_grad():
		if loader is not None:
			aTest(args, model, loader, noise_std, blind=blind, device=device, show=show, save=save)
		if dictionary:
			aDictionary(args, model, freq=freq, show=show, save=save)
		if passthrough:
			aPassthrough(args, model, img_path, noise_std, blind=blind, show=show, device=device, save=save)

def aTest(args, model, loader, noise_std=25, blind=False, device=torch.device('cpu'), show=False, save=True):
	""" Evaluate model on test-set.
	"""
	print("--------- test ---------")
	save_dir = args['paths']['save']
	if model.adaptive and blind in [True, 'MAD', 'wvlt']:
		fn = os.path.join(save_dir, "test_blindMAD.json")
	elif model.adaptive and blind == 'PCA':
		fn = os.path.join(save_dir, "test_blindPCA.json")
	else:
		fn = os.path.join(save_dir, "test.json")
	if os.path.exists(fn):
		fd = open(fn); log = json.load(fd); fd.close()
	else:
		log = {}
	if not type(noise_std) in [range, list, tuple]:
		noise_std = [noise_std]
	dset = loader.dataset.root_dirs[0]
	log[dset] = {}
	tracker = {}
	model.eval()
	for sigma in noise_std:
		tracker[str(sigma)] = {'psnr': []}
		tracker['times'] = []
		log[dset][str(sigma)] = {}
		t = tqdm(iter(loader), desc=f"TEST-{sigma}", dynamic_ncols=True)
		for itern, x in enumerate(t):
			if x.shape[2] > x.shape[3]: # for more consistent timing
				x = x.transpose(0,1)
			x = x.to(device)
			y, s = utils.awgn(x, sigma)
			with torch.no_grad():
				t0 = time.time()
				if model.adaptive:
					if blind:
						s = 255 * nle.noiseLevel(y, method=blind)
				else:
					s = None
				xhat, _ = model(y, s)
				t1 = time.time()
			tracker['times'].append(t1-t0)
			psnr = -10*np.log10(torch.mean((x-xhat)**2).item())
			tracker[str(sigma)]['psnr'].append(psnr)
		log[dset][str(sigma)] = {}
		log[dset][str(sigma)]['psnr-mean'] = np.mean(tracker[str(sigma)]['psnr'])
		log[dset][str(sigma)]['psnr-std']  =  np.std(tracker[str(sigma)]['psnr'])
		log[dset]['time-mean'] = np.mean(tracker['times'])
		log[dset]['time-std']  = np.std(tracker['times'])
	pprint(log)
	if save:
		print(f"Saving Testset log to {fn} ... ")
		with open(fn,'+a') as log_file:
			log_file.write(json.dumps(log, indent=4, sort_keys=True))

def aDictionary(args, model, freq=True, show=False, save=True):
	""" Saves model dictionary's filters, frequency-response.
	"""
	print("--------- dictionary ---------")
	figlist = [] # append figures so they may all be closed at the end
	save_dir = args['paths']['save']
	D = model.D.weight.cpu()
	# get effective dictionary
	n = int(np.ceil(np.sqrt(D.shape[0])))
	if show:
		fig = visplot(D, (D.shape[0]//n, n)); figlist.append(fig)
		fig.suptitle("Learned Dictionary")
	if save: # save dictionary as image and pytorch file
		fn = os.path.join(save_dir, "D_learned.png")
		print(f"Saving learned dictionary to {fn} ...")
		save_image(D, fn, nrow=n, padding=2, scale_each=True, normalize=True)
	# plot frequency response of effective dictionary
	if freq:
		X = torch.tensor(fftshift(fft2(D.detach().numpy(), (64,64)), axes=(-2,-1)))
		n = round(np.sqrt(D.shape[0]))
		if show:
			fig = visplot(X.abs(), (D.shape[0]//n, n), colorbar=False); figlist.append(fig)
			fig.suptitle("D Magniuted Response, (linear scale)")
			plt.show()
		if save:
			fn = os.path.join(save_dir, "freq.png")
			print(f"Saving dictionary magnitude response to {fn} ...")
			save_image(X.abs(), fn, nrow=n, normalize=True, scale_each=True, padding=10, pad_value=1)
	for fig in figlist:
		plt.close(fig)

def aPassthrough(args, model, img_path, noise_std, show=False, device=torch.device('cpu'), save=False, blind=False):
	""" Save passthrough of single image
	"""
	print("--------- passthrough ---------")
	figlist = []
	img_name = os.path.basename(img_path); img_name = os.path.splitext(img_name)[0]
	if save:
		save_dir = os.path.join(args['paths']['save'], f"passthrough_{img_name}")
		if not os.path.isdir(save_dir):
			os.mkdir(save_dir)
	print(f"using {img_path}...")
	x = utils.imgLoad(img_path, gray=True).to(device)
	y, sigma = utils.awgn(x, noise_std)
	print(f"noise_std = {sigma}")
	if model.adaptive:
		if blind:
			sigma = 255 * nle.noiseLevel(y, method=blind)
			print(f"sigma_hat = {sigma:.3f}")
		else:
			print(f"using GT sigma.")
	else:
		sigma = None
	xhat, csc = model(y, sigma)
	psnr = -10*np.log10(torch.mean((x-xhat)**2).item())
	print(f"PSNR = {psnr:.2f}")
	if not save and not show:
		return
	csc = csc.cpu().transpose(0,1); x=x.cpu(); y=y.cpu(); xhat = xhat.cpu()
	# image domain comparison
	images = torch.cat([y, xhat, x])
	fn = os.path.join(save_dir, f"compare.png")
	if save:
		print(f"Saving image domain comparison at {fn} ...")
		save_image(images, fn, padding=10, scale_each=False, normalize=False)
	if show:
		psnr_n = -10*np.log10(torch.mean((x-y)**2))
		fig = visplot(images, titles=[f'y, {psnr_n:.2f}', f'xhat, {psnr:.2f}', 'x']); figlist.append(fig)
	# csc
	fn = os.path.join(save_dir, f'csc.png')
	m = csc.abs().max()
	n = round(np.sqrt(csc.shape[0]))
	if save:
		print(f"Saving csc at {fn} ...")
		save_image(csc, fn, n, padding=10, scale_each=True, range=(-1,1))
	if show:
		fig = visplot(csc, (csc.shape[0]//n, n)); figlist.append(fig)
		fig.suptitle('csc')
	for fig in figlist:
		plt.close(fig)

if __name__ == "__main__":
	""" Load arguments from json file and command line and pass to main.
	"""
	default_cmds = ["--test=CBSD68","--noise_std=25","--passthrough", "--dictionary", "--img=Set12/09.png", "--blind=[False,\"PCA\",\"MAD\"]", "--show", "--save"]
	if len(sys.argv)<2 or not os.path.isfile(sys.argv[1]):
		print('ERROR: usage: analyze.py /path/to/arg_file.json [options]')
		for cmd in sys.argv[1:]:
			if cmd in ['--help', 'help', '-h']:
				print("[options]:", default_cmds)
				break
		sys.exit(1)
	args_file = open(sys.argv[1])
	args = json.load(args_file)
	pprint(args)
	args_file.close()
	# cli args
	cmds = sys.argv[2:]
	if '--help' in cmds or '-h' in cmds:
		print("[options]:", default_cmds); sys.exit()
	# Beware: ugly argument parsing...
	if "--test" in "".join(cmds):
		arg = "--test"
		for c in cmds:
			if "--test" in c:
				arg = c; break
		if len(arg.split("=")) == 1:
			raise ValueError("test set not provided (ex. --test=CBSD68).")
		else:
			tst_path = [arg.split("=")[1]]
		loader = data.getDataLoader(tst_path, load_color=False, test=True)
	else:
		loader = None
	arg = "--img"
	for c in cmds:
		if "--img" in c:
			arg = c; break
	if len(arg.split("=")) == 1:
		img_path = "Set12/09.png"
	else:
		img_path = arg.split("=")[1]
	if "--noise" in "".join(cmds):
		for c in cmds:
			if "--noise" in c:
				arg = c; break
		noise_std = eval(arg.split("=")[1])
	else:
		noise_std = 25
	print(f"noise_std = {noise_std}")
	if "--blind" in "".join(cmds):
		arg = "--blind"
		for c in cmds:
			if "--blind" in c:
				arg = c; break
		if len(arg.split("=")) == 1:
			blind = True
		else:
			blind = eval(arg.split("=")[1])
	else:
		blind = False
	show = True if "--show" in cmds else False
	save = True if "--save" in cmds else False
	freq = True if "--freq" in cmds else False
	pt   = True if "--passthrough" in cmds else False
	dictionary = True if "--dictionary" in cmds else False
	# run analysis 
	main(args, img=img_path, noise_std=noise_std, dictionary=dictionary, passthrough=pt, loader=loader, blind=blind, save=save, show=show, freq=freq)

