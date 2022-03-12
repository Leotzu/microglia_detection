# Combine .png tiles into a zarr array for use in gunpowder

import os
import glob
import numpy as np
import zarr
from PIL import Image
from sklearn.model_selection import train_test_split

# Relevant directories
tiles_dir = 'tiles'
input_dir = os.path.join(tiles_dir, 'input')
dots_dir  = os.path.join(tiles_dir, 'label_dot')
cells_dir = os.path.join(tiles_dir, 'label_cell-body')

# Output names
output_zarr = 'train-test-split.zarr'

# Get list of images from the input directory.
filenames = glob.glob(os.path.join(input_dir, '*.png'))
# Remove the directory prefix from each filename.
filenames = [os.path.split(x)[-1] for x in filenames]

store = zarr.DirectoryStore(output_zarr)
root = zarr.group(store=store)
data = []
for f in filenames:
    # Read each image. The raw will have an alpha channel by default - drop it.
    # Dots and cells will only have a single channel, no need to convert.
    raw   = np.array(Image.open(os.path.join(input_dir, f)).convert(mode='RGB')).transpose(2,0,1)
    dots  = np.array(Image.open(os.path.join(dots_dir, f)))
    cells = np.array(Image.open(os.path.join(cells_dir, f)))

    # Combine the above datasets
    combined_arr = np.insert(np.insert(raw, 3, dots, axis=0), 4, cells, axis=0)
    data.append(combined_arr)

# Sample out a 20% test set
train, test = train_test_split(np.stack(data), test_size=0.2)

# Write the data to zarr
root.array('train', train)
root.array('test', test)
