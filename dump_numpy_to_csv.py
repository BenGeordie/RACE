import numpy as np
import sys

data = np.load(sys.argv[1])
x_cat = data['X_cat'].astype(np.int32)
x_int = data['X_int'].astype(np.int32)
y = np.reshape(data['y'], newshape=(-1, 1))
data.close()
data = np.concatenate([y, x_int, x_cat], axis=1)
np.savetxt(sys.argv[2], data, fmt='%1d', delimiter=',')