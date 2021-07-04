#!/usr/bin/env python3
import os, sys, json, glob
import matplotlib.pyplot as plt
import torch

flist = glob.glob("Models/CDLNet-color_AB-[2-3]a/test.json")
namelist = []
tstlist = []

#single_psnrs = [37.95, 33.89, 31.74, 30.31, 29.26, 28.45, 27.78, 27.22, 26.75, 26.35]; 
#DnCNN_psnrs = [33.70, 32.90, 31.57, 30.17, 29.14, 28.31, 27.66, 25.92, 22.70, 20.15];
#FFDNet_psnrs = [36.22, 33.48, 31.50, 30.13, 29.10, 28.29, 27.63, 27.07, 26.58, 26.15]; 
noise_CDnCNN = [5, 10, 15, 20, 25]
CDnCNNB = [40.24, 35.88, 33.49, 31.88, 30.68]
noise_S = [5,15,25,35,45,50]
Big_CCDLNet_S = [40.48, 34.03, 31.37, 29.75, 28.61, 28.15]

for f in flist:
	namelist.append(f[-12:-10])
	fp = open(f)
	tstlist.append(json.load(fp))
	fp.close()

fig = plt.figure()
marker = ['-or', '-ob']
for i in range(len(flist)):
	name = namelist[i]
	tst = tstlist[i]
	keys = tst['CBSD68'].keys()
	sigma = []
	psnr = []
	for k in keys:
		if "time" in k or "0.001" in k:
			continue
		sigma.append(int(k))
		psnr.append(tst['CBSD68'][k]['psnr-mean'])
	first = sigma.pop(-2)
	firstpsnr  = psnr.pop(-2)
	sigma = [first, *sigma]
	psnr = [firstpsnr, *psnr]

	plt.plot(sigma, psnr, marker[i], label=name)

plt.plot(noise_S, Big_CCDLNet_S, "g*", mec='k', label="Big-CCDLNet-S", ms=10)
plt.axvspan(0, 55, color='blue', alpha=0.1, label=r"$\sigma_n^{\mathrm{train}} = [0,55]$")
#plt.plot(noise_CDnCNN, CDnCNNB, label="CDnCNN-B")
#plt.plot(sigma, FFDNet_psnrs, label="FFDNet*")
plt.legend()
plt.grid()

plt.show()
