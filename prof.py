import torch
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as modelsdd
from net import CDLNet
from data import getFitLoaders
import sys
import json
import os 

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


args_file = open(sys.argv[1])
args = json.load(args_file)
args_file.close()
ngpu = torch.cuda.device_count()
device = torch.device("cuda:0" if ngpu > 0 else "cpu")
model_args, train_args, paths = [args[item] for item in ['model','train','paths']]
loaders = getFitLoaders(**train_args['loaders'])
train_loader = loaders['train']
model, opt, sched, epoch0 = initModel(args, device=device)

model = model.to(device)
for i, batch in enumerate(train_loader):
    batch = batch.to(device)
    model(batch)

with profile(activities=[
    ProfilerActivity.CPU,
    ProfilerActivity.CUDA
    ], profile_memory = False, record_shapes=True) as prof:
    with record_function("model_inference"):
        for i, batch in enumerate(train_loader):
            batch = batch.to(device)
            model(batch)

#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
#print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
