# Converts .tif files to np.arrays then stores them in a zarr container for training
# Images are stored in 01_data/<data_name>/trainingset_xx/zarr
# Once in a zarr, training can be done from 02_train/setup_xx/train.py

import matplotlib.pyplot as plt
import numpy as np
import zarr
from PIL import Image

# Change this to your target data folder
data_dir = '008/trainingset_01/'

# convert raw and groundtruth images into np.arrays
raw_raster = Image.open(data_dir + 'raw.tif')
seg_raster = Image.open(data_dir + 'seg.tif')

# transpose data so it's in the form: (channels, x, y)
raw_data = np.array(raw_raster).transpose(2,0,1) 

seg_data = np.array(seg_raster)
seg_data = seg_data[np.newaxis,:].astype(np.float32) #Q: why float32??

# store the images in a zarr container
f = zarr.open(data_dir + 'zarr.zarr', 'w')
f['raw'] = raw_data
f['raw'].attrs['resolution'] = (1, 1)
f['seg'] = seg_data
f['seg'].attrs['resolution'] = (1, 1)
