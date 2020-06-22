import torch
import time

GPU = True
INFINITY = False

# preload the matrices
print("Preloading matrices...", flush=True)
mats = {}
MATSZ = [100, 500, 1000, 2000, 4000, 8000, 16000, 32000]
for msz in MATSZ:
    mat = torch.rand(msz, msz)
    if GPU:
        mats[msz] = mat.cuda()
    else:
        mats[msz] = mat
print("Done preloading.", flush=True)

def run_benchmark(msz):
    ops = 2*msz**3
    start = time.time()
    mat = mats[msz]
    r = torch.mm(mat, mat).cuda()
    if GPU:
        torch.cuda.synchronize()
    end = time.time()
    duration = end - start
    tflops = ops / duration / 10**12
    print("{}x{} MM {} ops in {} sec = TFLOPS {}"
          .format(msz, msz, ops, duration, tflops),
          flush=True)

while True:
    print("======== Results ======== ", flush=True)
    for msz in MATSZ:
        run_benchmark(msz)
    if not INFINITY:
        break