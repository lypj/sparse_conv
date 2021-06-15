#!/usr/bin/env python3
import sys, json
from os.path import join
from pprint import pprint
import numpy as np

def write_args(arg_dict, name):
	with open(join("args.d",name+".json"), '+w') as outfile:
		outfile.write(json.dumps(arg_dict, indent=4, sort_keys=True))

args_file = open("args.json")
args = json.load(args_file)
args_file.close()

loop_args = {
	"noise_std": [5,15,25,35,45,50],
}

args["model"] = {
	"adaptive": False,
	"num_filters": 100,
	"num_inchans": 3,
	"filter_size": 7,
	"stride": 2,
	"iters": 22,
	"tau0": 1e-2,
}

args["train"] = {
	"loaders": {
		"batch_size": 10,
		"crop_size": 128,
		"load_color": True,
		"trn_path_list": ["CBSD432"],
		"val_path_list": ["Kodak"],
		"tst_path_list": ["CBSD68"]
	},
	"fit": {
		"epochs": 6000,
		"noise_std": 25,
		"val_freq": 50,
		"save_freq": 5,
		"backtrack_thresh": 2,
		"verbose": False,
		"clip_grad": 5e-2,
	},
	"opt": {
		"lr": 5e-4
	},
	"sched": {
		"gamma": 0.95,
		"step_size": 50
	}
}

args['type'] = "CDLNet"
#args['paths']['ckpt'] = ""
#epoch0 = "4000.ckpt"
#ckpt = "Models/CDLNet-nht_trnweight-0a/4000.ckpt"
vnum = 0
name = "color_S"

def product(*args, repeat=1):
	# product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
	# product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
	pools = [tuple(pool) for pool in args] * repeat
	result = [[]]
	for pool in pools:
		result = [x+[y] for x in result for y in pool]
	for prod in result:
		yield tuple(prod)

keys = list(loop_args.keys())

with open(f"Models/{args['type']}-{name}.summary", "a") as summary:
	for items in product(*[loop_args[k] for k in keys]):
		for i, it in enumerate(items):
			if keys[i] in args['model']:
				args['model'][keys[i]] = it
			elif keys[i] in args['train']:
				args['train'][keys[i]] = it
			elif keys[i] in args['train']['fit']:
				args['train']['fit'][keys[i]] = it
			elif keys[i] in args['train']['loaders']:
				args['train']['loaders'][keys[i]] = it

		version = args['type']+"-" + name + "-" + str(vnum)
		args['paths']['save'] = "Models/" + version
		if args['paths']['ckpt'] is not None:
			#args['paths']['ckpt'] = ckpt + str(vnum) + "a/" + epoch0
			args['paths']['ckpt'] = ckpt
		write_args(args, version)
		print(f'{version}: {items}')
		summary.write(f'{version}: {items}\n')
		vnum += 1

