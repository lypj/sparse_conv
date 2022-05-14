Deeping Convolutional Dictionary Learning with Sparse Convolution

Cloned from https://github.com/nikopj/CDLNet
Modified with https://github.com/traveller59/spconv

Requirements: 
matplotlib==3.3.3
numpy==1.19.5
Pillow==8.1.0
PyWavelets==1.1.1
scipy==1.6.0
torch==1.7.1+cu110
torchvision==0.8.2+cu110
tqdm==4.55.1
spconv==2.1(pip install spconv-cu113)

Run: 
python train.py args.json

Key parameters in args.json used in experiments:
num_filters
iters
tau0
sp_conv
batch_size
fp16


