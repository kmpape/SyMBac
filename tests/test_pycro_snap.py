# simple single image acquisition example with snap
from pycromanager import Core
import numpy as np
import matplotlib.pyplot as plt
import time

#### Setup ####
core = Core()

#### imaging settings
exposure = 20
num_frames_to_capture = 100
core.set_exposure(exposure)

interframe_interval = 10000 # ms
assert interframe_interval > exposure


frames = []
model = []
acq_timestamps = []
process_timestamps = []

core.snap_image()
tagged_image = core.get_tagged_image()
tmp = np.reshape(tagged_image.pix, newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
fig = plt.figure(figsize=[tagged_image.tags['Height'], tagged_image.tags['Width']]) # initialize figure
plt.subplot(1, 1, 1)
plt.imshow(tmp, cmap='gray', vmin=0, vmax=65535)
plt.axis('off')
plt.show(block=False)

print('beginning acquisition...')
t0 = time.time()
next_call = time.time()  # updated periodically for when to take next image
for f in range(num_frames_to_capture):

    # snap image
    tagged_image = core.get_tagged_image()

    plt.close()
    plt.imshow(np.reshape(tagged_image.pix,
                          newshape=[tagged_image.tags['Height'],
                                    tagged_image.tags['Width']]),
               cmap='gray', vmin=0, vmax=65535)
    plt.axis('off')
    plt.show(block=False)

    # helpful printout to monitor progress
    print('current frame: {}'.format(f))

    nowtime = time.time()
    next_call = next_call + interframe_interval / 1000
    if next_call - nowtime < 0:
        print("warning: strobe delay exceeded inter-frame-interval on frame {}.".format(f))
    else:
        time.sleep(next_call - nowtime)

print('done!')