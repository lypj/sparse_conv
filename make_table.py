#!/usr/bin/env python3
import os, sys, json, glob,  copy
import re # regular expressions
from glob import glob
import numpy as np
model_name = "Bigger"
model_dirs = glob(f'./Models/CDLNet-{model_name}-[0-3]a/')

tf = open('table.txt','w')

def load_json(path):
	f = open(path,'r')
	obj = json.load(f)
	f.close()
	return obj


tf.write("\\begin{tabular}{cccccc}\n")
tf.write("Model \\# & Params & K & M & Stride & O(FLOPS) & $\sigma_n=25$ \\\\ \\hline\n")

for d in sorted(model_dirs):
	d = copy.copy(d[:-2]) + "a"
	name  = re.sub(f'./Models/CDLNet-','',d)[:-1]

	db = copy.copy(d[:-1]) + "b"
	nameb  = re.sub(f'./Models/CDLNet-','',db)[:-1]

	# LOAD ARGS
	fn = os.path.join(d,'args.json')
	fnb = os.path.join(db,'args.json')
	if os.path.exists(fn):
		args = load_json(fn)
	elif os.path.exists(fnb):
		args = load_json(fnb)
	else:
		continue

	# LOAD TEST.JSON
	# fn = os.path.join(d,'test.json')
	# if not os.path.exists(fn):
	# 	continue
	# tst  = load_json(os.path.join(d,'test.json'))
	# psnr = [tst['CBSD68'][str(p)]['psnr-mean'] for p in bern_p]

	# fn = os.path.join(db,'test.json')
	# if not os.path.exists(fn):
	# 	psnrb = -1
	# else:
	# 	tstb  = load_json(os.path.join(db,'test.json'))
	# 	psnrb = [tstb['CBSD68'][str(p)]['psnr-mean'] for p in bern_p]

	# LOAD TEST.PSNR
	fn = os.path.join(d,'test.psnr')
	if not os.path.exists(fn):
		psnr = "-"
	else:
		psnr  = float(np.loadtxt(fn))
		psnr = f"{psnr:.2f}"
	fn = os.path.join(db,'test.psnr')
	if not os.path.exists(fn):
		psnrb = "-"
	else:
		psnrb  = float(np.loadtxt(fn))
		psnrb = f"{psnrb:.2f}"

	# LOAD BACKTRACKING
	fn = os.path.join(d,'backtrack.txt')
	if os.path.exists(fn):
		bt = len(np.atleast_1d(np.loadtxt(fn, delimiter=None)))
	else:
		bt = 0

	fn = os.path.join(db,'backtrack.txt')
	if os.path.exists(fn):
		btb = len(np.atleast_1d(np.loadtxt(fn, delimiter=None)))
	else:
		btb = 0

	print(d)
	print(db)

	M = args['model']['num_filters']
	K = args['model']['iters']
	#C = args['model']['num_inchans']
	C = 1
	ks= args['model']['filter_size']
	stride= args['model']['stride']
	#meansub = ''.join([str(z) for z in args['model']['meansub']])
	ada = args['model']['adaptive']
	bs  = args['train']['loaders']['batch_size']
	cs  = args['train']['loaders']['crop_size']

	params = int( ((K-1)*( 2*C*M*(ks**2) + M ) + 2*C*M*(ks**2) + M) / 1e3 )
	flops  = int( (K*C*M*ks**2 /stride**2) / 1e3)

	#psnr_str = "&".join([f" {psnr[i]:.2f}/{psnrb[i]:.2f} " for i in range(len(psnr))])

	tf.write(f"{name}/b & {params}k & {K} & {M} & {stride} & {flops}k & "+ psnr +"/" +psnrb+ " \\\\ \n")

tf.write("\\hline \n\\end{tabular}")
tf.close()


