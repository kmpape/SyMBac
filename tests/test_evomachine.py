import sys
# sys.path.insert(1, '/home/idris/workspace_python/symbac/SyMBac/') # Not needed if you installed SyMBac using pip

import cv2
import numpy as np
import matplotlib.pyplot as plt

from SyMBac.simulation import Simulation
from SyMBac.PSF import PSF_generator
from SyMBac.renderer import Renderer
from SyMBac.PSF import Camera
from SyMBac.misc import get_sample_images


# OPTIONS
show_window = False
# save_dir = "/home/idris/workspace_python/symbac/testdata/"
save_dir = "/home/hslab/workspace_python/symbac_pip/testdata/"
save_mask = save_dir + "tmp_mask.p"
param_set = 1  # 0= SyMBac, 1= EvoMachine

# TRENCH
if param_set == 0:
    trench_length = 15  # um
    trench_width = 1.3  # um
else:
    trench_length = 50  # um
    trench_width = 3  # um

# CELLS
cell_max_length = 6.65  # um
cell_width = 0.9  # um
max_length_var = 0.  # um, NOTE: all _var are in fact STDs
width_var = 0.  # um
lysis_p = 0.  # [0,1]

# SIMULATION
sim_length = 50
gravity = 0.1
phys_iters = 15

# PSF
mode = "simple fluo"
radius = 50  # Radius of the PSF                                        -> unclear
wavelength = 0.65  # Wavelength of imaging light in micron              -> 450nm-650nm, MH12
num_ap = 0.95  # Numerical aperture of the objective lens               -> 0.95 or 1.42, MH4
ref_ind = 1.3  # Refractive index of the imaging medium                 -> unclear
resize_amount = 3  # Upscaling factor, typically chosen to be 3         -> Upscaling before convolution
if param_set == 0:
    pix_mic_conv = 0.065
else:
    pix_mic_conv = 0.1625  # Micron per pixel conversion factor             -> 6.5/40, MH4

# EXAMPLE IMAGE
tmp = get_sample_images()["E. coli 100x"]  # size must somehow match the output of the simulation
if param_set == 1:
    num_rows, num_cols = tmp.shape
    div_factor = pix_mic_conv / 0.065
    real_image = cv2.resize(tmp, dsize=(int(tmp.shape[1]/div_factor), int(tmp.shape[0]/div_factor)),
                            interpolation=cv2.INTER_CUBIC)
else:
    real_image = tmp

# CAMERA
baseline = 100
sensitivity = 2.9
dark_noise = 8

my_simulation = Simulation(
    trench_length=trench_length, trench_width=trench_width,
    cell_max_length=cell_max_length, cell_width=cell_width,
    max_length_var=max_length_var, width_var=width_var, lysis_p=lysis_p,
    pix_mic_conv=pix_mic_conv, resize_amount=resize_amount,
    gravity=gravity, phys_iters=phys_iters, sim_length=sim_length, save_dir=save_dir
)
my_simulation.run_simulation(show_window=show_window)
my_simulation.draw_simulation_OPL(do_transformation=True, label_masks=True)
if show_window:
    my_simulation.visualise_in_napari()

my_camera = Camera(
    baseline=baseline, sensitivity=sensitivity, dark_noise=dark_noise
)
if show_window:
    my_camera.render_dark_image(size=(300, 300))

my_kernel = PSF_generator(
    mode="simple fluo", radius=radius, wavelength=wavelength, NA=num_ap, n=ref_ind,
    resize_amount=resize_amount, pix_mic_conv=pix_mic_conv, apo_sigma=0.
)
my_kernel.calculate_PSF()
if show_window:
    my_kernel.plot_PSF()
    input("Press Enter to continue...")

my_renderer = Renderer(
    simulation=my_simulation, PSF=my_kernel, real_image=real_image, camera=my_camera
)

my_renderer.select_intensity_napari(fname=save_mask)
my_renderer.optimise_synth_image(manual_update=False)

