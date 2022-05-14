#!/usr/bin/env python3
""" CDLNet/train.py
Executable for training networks. Also defines training loop and associated functions.
Usage: $./train.py /path/to/args.json
"""
import os, sys, json
from tqdm import tqdm
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
from net import CDLNet
from data import getFitLoaders
from utils import awgn
from torch.profiler import profile, record_function, ProfilerActivity
import contextlib
import time

def main(args):
	""" Given argument dictionary, load data, initialize model, and fit model.
	"""
	ngpu = torch.cuda.device_count()
	device = torch.device("cuda:0" if ngpu > 0 else "cpu")
	print(f"Using device {device}.")
	if ngpu > 1:
		print(f"Using {ngpu} GPUs.")
		data_parallel = True
	else:
		data_parallel = False
	model_args, train_args, paths = [args[item] for item in ['model','train','paths']]
	loaders = getFitLoaders(**train_args['loaders'])
	model, opt, sched, epoch0 = initModel(args, device=device)
	fit(model, opt, loaders,
	    sched       = sched,
	    save_dir    = paths['save'],
	    start_epoch = epoch0 + 1,
	    device      = device,
	    **train_args['fit'],
	    epoch_fun = lambda epoch_num: saveArgs(args, epoch_num))

def fit(model, opt, loaders,
	    sched = None,
	    epochs = 1,
	    device = torch.device("cpu"),
	    save_dir = None,
	    start_epoch = 1,
	    clip_grad = 1,
	    noise_std = 25,
	    verbose = True,
	    val_freq  = 1,
	    save_freq = 1,
	    fp16 = False,
	    epoch_fun = None,
	    backtrack_thresh = 1):
	""" fit model to training data.
	"""
	if not type(noise_std) in [list, tuple]:
		noise_std = (noise_std, noise_std)
	print(f"fit: using device {device}")
	print("Saving initialization to 0.ckpt")
	path = os.path.join(save_dir, '0.ckpt')
	saveCkpt(path, model, 0, opt, sched)
	top_psnr = {"train": 0, "val": 0, "test": 0} # for backtracking
	epoch = start_epoch
	total_time = 0
	while epoch < start_epoch + epochs:
		for phase in ['train', 'val', 'test']:
			model.train() if phase == 'train' else model.eval()
			if epoch != epochs and phase == 'test':
				continue
			if phase == 'val' and epoch%val_freq != 0:
				continue
			if phase in ['val', 'test']:
				phase_nstd = (noise_std[0]+noise_std[1])/2.0
			else:
				phase_nstd = noise_std
			psnr = 0
			scaler = torch.cuda.amp.grad_scaler.GradScaler()
			amp = contextlib.nullcontext()
			if fp16: 
				amp =  torch.cuda.amp.autocast()
			t = tqdm(iter(loaders[phase]), desc=phase.upper()+'-E'+str(epoch), dynamic_ncols=True)
		#	with profile(activities=[
		#		ProfilerActivity.CPU,
		#		ProfilerActivity.CUDA
		#		], profile_memory = True, record_shapes=True) as prof:
		#		with record_function("model_inference"):

			start = time.perf_counter()
			for itern, batch in enumerate(t):
				batch = batch.to(device)
				noisy_batch, sigma_n = awgn(batch, phase_nstd)
				opt.zero_grad()
				with torch.set_grad_enabled(phase == 'train'):
					with amp:
						batch_hat, _ = model(noisy_batch, sigma_n)
						loss = torch.mean((batch - batch_hat)**2)
						scale = 1.0
						if phase == 'train':
							if fp16:
								scaler.scale(loss).backward()
							else:
								loss.backward()
							if clip_grad is not None:
								nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
							if fp16:
								scaler.step(opt)
								scaler.update()
								scale = scaler.get_scale()
							else:
								opt.step()
							model.project()
				loss = loss.item()
				if verbose:
					total_norm = grad_norm(model.parameters())
					t.set_postfix_str(f"loss={loss:.1e}|gnorm={total_norm:.1e}")
				psnr = psnr - 10*np.log10(loss)
						
			torch.cuda.synchronize()
			end = time.perf_counter()
			total_time = total_time + end - start
			print(f"Total time: {end-start:3f}s")
		#	print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
			psnr = psnr/(itern+1)
			if psnr > 27:
				print(f"TTA 27: {total_time:3f}s")
			print(f"{phase.upper()} PSNR: {psnr:.3f} dB")
			if psnr > top_psnr[phase]:
				top_psnr[phase] = psnr
			# backtracking check
			elif (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
				break
			with open(os.path.join(save_dir, f'{phase}.psnr'),'a') as psnr_file:
				psnr_file.write(f'{psnr:.3f}  ')
		if (psnr + backtrack_thresh < top_psnr[phase]) or np.isnan(loss) or np.isinf(loss):
			with open(os.path.join(save_dir, f'backtrack.txt'),'a') as psnr_file:
				psnr_file.write(f'{epoch}  ')
			if epoch % save_freq == 0:
				epoch = epoch - save_freq
			else:
				epoch = epoch - epoch%save_freq
			old_lr = np.array(getlr(opt))
			print(f"Model has diverged. Backtracking to {epoch}.ckpt ...")
			path = os.path.join(save_dir, str(epoch) + '.ckpt')
			model, _, _, _ = loadCkpt(path, model, opt, sched)
			new_lr = old_lr * 0.8
			print("Updated Learning Rate(s):", new_lr)
			setlr(opt, new_lr)
			epoch = epoch + 1
			continue
		if sched is not None:
			sched.step()
			if hasattr(sched, "step_size") and epoch % sched.step_size == 0:
				print("Updated Learning Rate(s): ")
				print(getlr(opt))
		if epoch % save_freq == 0:
			path = os.path.join(save_dir, str(epoch) + '.ckpt')
			print('Checkpoint: ' + path)
			saveCkpt(path, model, epoch, opt, sched)
			if epoch_fun is not None:
				epoch_fun(epoch)
		epoch = epoch + 1

def grad_norm(params):
	""" computes norm of mini-batch gradient
	"""
	total_norm = 0
	for p in params:
		param_norm = torch.tensor(0)
		if p.grad is not None:
			param_norm = p.grad.data.norm(2)
		total_norm = total_norm + param_norm.item()**2
	return total_norm**(.5)

def getlr(opt):
	return [pg['lr'] for pg in opt.param_groups]

def setlr(opt, lr):
	if not issubclass(type(lr), (list, np.ndarray)):
		lr = [lr for _ in range(len(opt.param_groups))]
	for (i, pg) in enumerate(opt.param_groups):
		pg['lr'] = lr[i]

def initModel(args, device=torch.device("cpu")):
	""" Return model, optimizer, scheduler with optional initialization
	from checkpoint.
	"""
	model_type, model_args, paths = [args[item] for item in ['type','model','paths']]
	train = False
	if 'train' in args:
		train_args = args['train']
		train = True
	if paths['ckpt'] is not None:
		init = False
	else:
		init = True 
	if model_type == "CDLNet":
		model = CDLNet(**model_args, init=init)
	else:
		raise NotImplementedError
	model.to(device)
	initDir = lambda p: os.mkdir(p) if not os.path.isdir(p) else None
	initDir(os.path.dirname(paths['save']))
	initDir(paths['save'])
	opt = torch.optim.Adam(model.parameters(), **train_args['opt'])     if train else None
	sched = torch.optim.lr_scheduler.StepLR(opt, **train_args['sched']) if train else None
	ckpt_path = paths['ckpt']
	if ckpt_path is not None:
		print(f"Initializing model from {ckpt_path} ...")
		model, opt, sched, epoch0 = loadCkpt(ckpt_path, model, opt, sched)
	else:
		epoch0 = 0
	if train:
		print("Current Learning Rate(s):")
		for param_group in opt.param_groups:
			print(param_group['lr'])
	total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Total Number of Parameters: {total_params:,}")
	return model, opt, sched, epoch0

def saveCkpt(path, model=None,epoch=None,opt=None,sched=None):
	""" Save Checkpoint.
	Saves model, optimizer, scheduler state dicts and epoch num to path.
	"""
	getSD = lambda obj: obj.state_dict() if obj is not None else None
	torch.save({'epoch': epoch,
	            'model_state_dict': getSD(model),
	            'opt_state_dict':   getSD(opt),
	            'sched_state_dict': getSD(sched)
	            }, path)

def loadCkpt(path, model=None,opt=None,sched=None):
	""" Load Checkpoint.
	Loads model, optimizer, scheduler and epoch number
	from state dict stored in path.
	"""
	ckpt = torch.load(path, map_location=torch.device('cpu'))
	def setSD(obj, name):
		if obj is not None and name+"_state_dict" in ckpt:
			print(f"Loading {name} state-dict...")
			obj.load_state_dict(ckpt[name+"_state_dict"])
		return obj
	model = setSD(model, 'model')
	opt   = setSD(opt, 'opt')
	sched = setSD(sched, 'sched')
	return model, opt, sched, ckpt['epoch']

def saveArgs(args, epoch_num=None):
	""" Write argument dictionary to file,
	with optionally writing the checkpoint.
	"""
	save_path = args['paths']['save']
	if epoch_num is not None:
		ckpt_path = os.path.join(save_path, f"{epoch_num}.ckpt")
		args['paths']['ckpt'] = ckpt_path
	with open(os.path.join(save_path, "args.json"), "+w") as outfile:
		outfile.write(json.dumps(args, indent=4, sort_keys=True))

if __name__ == "__main__":
	""" Load arguments dictionary from json file to pass to main.
	"""
	if len(sys.argv)<2:
		print('ERROR: usage: train.py [path/to/arg_file.json]')
		sys.exit(1)
	args_file = open(sys.argv[1])
	args = json.load(args_file)
	pprint(args)
	args_file.close()
	main(args)

