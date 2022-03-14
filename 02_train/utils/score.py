from scipy import ndimage
import numpy as np

# Counts the number of objects (clumps) in a 2d numpy array
# pixel threshold is pixel intensity to consider as an object
# size threshold is the size of object to consider an object (ie 1 pixel might be too small to consider)
def count_objects(data, pixel_threshold=0.7, size_threshold=1):
    
    data[data > pixel_threshold] = 1
    data[data <= pixel_threshold] = 0
    
    label, num_label = ndimage.label(data == 1)
    size = np.bincount(label.ravel())
    num_objects = (size > size_threshold).sum()
    
    return num_objects

def count_xor_difference(arr1, arr2, threshold=0.7):
    """Count the number of pixels that differ between two numpy arrays.

    arr1,arr2 -- arrays to compare
    threshold -- array elements above this threshold will be considered
                 'on', and 'off' if below or equal to.
    """

    return np.count_nonzero(np.logical_xor(arr1 > threshold,
                                           arr2 > threshold))
