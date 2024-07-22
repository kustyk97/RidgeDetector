import numpy as np

data  = np.load("InterpolatedData.npy")
data = data[2000:4000,1000:3000]
np.save("CropData.npy", data)