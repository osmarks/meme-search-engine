import h5py
import numpy as np
f = h5py.File("glove-200-angular.hdf5", "r")
data = np.array(f["train"])
norms = np.linalg.norm(data, axis=1)
data = data / norms[:, np.newaxis]
data.astype(np.float16).tofile("glove-200-angular.bin")
