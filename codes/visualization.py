import os
from vedo import Volume, show
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "../output", "EV_bspline.npy")

raw = np.load(data_path)
sample = (raw - raw.min()) / (raw.max() - raw.min())

volume = Volume(sample)

volume.cmap("gray")
volume.alpha([(0.2, 0), (0.6, 0.9)])

show(volume, axes=1, viewup='y', bg='black', title="EV")