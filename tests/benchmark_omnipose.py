from glob import glob
from natsort import natsorted
import numpy as np
import os
import psutil
import tifffile
from time import perf_counter

from cellpose import models
from omnipose.utils import normalize99

process = psutil.Process(os.getpid())  # Set highest priority for the python script for the CPU
process.nice(psutil.IOPRIO_CLASS_RT)

save_dir = "/home/hslab/workspace_python/symbac_pip/testdata/"
model_dir = save_dir + "train/models/"
convs_dir = save_dir + "convolutions/"
test_dir = save_dir + "large/"
convs = sorted(glob(convs_dir + "/*"))
first_img = tifffile.imread(convs[0])
img_shape = first_img.shape  # 102, 18
evo_shape = (3200, 3200)
num_tiles = (int(evo_shape[0] / img_shape[0]), int(evo_shape[1] / img_shape[1]))
num_imgs = num_tiles[0] * num_tiles[1]
print("num imgs={}".format(num_imgs))
large_img = np.zeros(evo_shape, dtype=np.float64)

irow = 0
for i, fname in enumerate(convs[:num_imgs]):
    irow = int(i / num_tiles[1])
    icol = np.mod(i, num_tiles[1])
    img_i = tifffile.imread(fname)
    img_i = img_i / img_i.max()
    large_img[irow * img_shape[0]:(irow + 1) * img_shape[0], icol * img_shape[1]:(icol + 1) * img_shape[1]] = img_i
    # Below is for testing segmentation on smaller bits but the 3200 x 3200 image should fit into RAM
    # if irow != int(i / num_tiles[1]):
    #    img_bench = copy.deepcopy(large_img[(irow-1) * img_shape[0]:irow * img_shape[0], :])
    #    start = perf_counter()
    #    elapsed = perf_counter() - start

model_list = natsorted(glob(save_dir + "train/models/*"))
model_name = model_list[-1]
print(model_name)
model = models.CellposeModel(gpu=True, pretrained_model=model_name, omni=True, concatenation=True)

start = perf_counter()
large_img = normalize99(large_img)  # required preprocessing
elapsed = perf_counter() - start
print("Preprocessing. Elapsed={:.4f}".format(elapsed))

nruns = 10
model_loaded = True
loop_run = True

omni_vec = [True]
batch_size_vec = [8]
irow_vec = range(0, num_tiles[0])

assert nruns > 2
omni = omni_vec[0]
batch_size = batch_size_vec[0]
#for omni in omni_vec:
#    for batch_size in batch_size_vec:  # number of 224 x 224 batches they send to the GPU
for irow in irow_vec:
    elapsed = np.zeros(nruns)
    x = [large_img[irow * img_shape[0]:(irow + 1) * img_shape[0], :]]
    for irun in range(0, nruns):
        start = perf_counter()
        masks, flows, styles = model.eval(
            x=x, batch_size=batch_size, channels=[0, 0], rescale=None, mask_threshold=-1,
            transparency=True, flow_threshold=0., omni=omni, resample=True, verbose=0)
        elapsed[irun] = perf_counter() - start
    avg_elapsed = (elapsed.sum()-elapsed.min()-elapsed.max())/(nruns-2.)
    print("Omni={:1}\tBatch Size={:2}\tnrows={:2}\tElapsed_s={:.4f}\tMin_s={:.4f}\tMax_s={:.4f}\tElapsedPerTrench_ms={:.4f}\t".format(
        omni, batch_size, irow+1, avg_elapsed, elapsed.min(), elapsed.max(), avg_elapsed/(irow+1)/num_tiles[1]*1000))
