import torch 
import spconv.pytorch as spconv
import time

x = torch.zeros((20,1,1280,1280))
    
for i in range(10):
    for j in range(1280):
        x[i][0][j][j] = 1.0
x = x.to(torch.device('cuda'))

m = torch.nn.Conv2d(1,1,5)
m = m.to(torch.device('cuda'))
start = time.perf_counter()
for i in range(10):
    y = m(x)
torch.cuda.synchronize()
end = time.perf_counter()
print("Time:",end - start)


x_sp = spconv.SparseConvTensor.from_dense(x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
print(x_sp.features)
#sp_m = spconv.SubMConv2d(1, 1, 5, 1, algo=spconv.core.ConvAlgo.MaskSplitImplicitGemm)
sp_m = spconv.SparseSequential(
    spconv.SubMConv2d(1, 1, 5, 1, algo=spconv.core.ConvAlgo.Native),
    spconv.ToDense()
)
sp_m = sp_m.to(torch.device('cuda'))
start1 = time.perf_counter()
for i in range(10):
    y_sp = sp_m(x_sp)
torch.cuda.synchronize()
end1 = time.perf_counter()
print("Time:",end1 - start1)
