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

# Computes the distances between the centroids of the predicted
# objects to the centroids of the actual objects
# Returns a list of distances
def compute_centroids(seg, pred, pixel_threshold=0.7, size_threshold=1):
    
    seg_coordinates_dict = {}
    pred_coordinates_dict = {}
    
    seg_centroid_dict = {}
    pred_centroid_dict = {}
    
    # Predicted Objects
    pred[pred > pixel_threshold] = 1
    pred[pred <= pixel_threshold] = 0
    
    label, num_label = ndimage.label(pred == 1)
    size = np.bincount(label.ravel())
    num_objects = (size > size_threshold).sum()
    
    for i in range(1,len(size)):
        pairs_pred = np.asarray(np.where(label == i)).T
        pred_coordinates_dict[i] = pairs_pred
        
    for key,value in pred_coordinates_dict.items():
        x2 = [p[0] for p in value]
        y2 = [p[1] for p in value]
        centroid_pred = (sum(x2) / len(value), sum(y2) / len(value))
        pred_centroid_dict[key] = centroid_pred
        
    # Seg Objects
    seg[seg > pixel_threshold] = 1
    seg[seg <= pixel_threshold] = 0
    
    label, num_label = ndimage.label(seg == 1)
    size = np.bincount(label.ravel())
    num_objects = (size > size_threshold).sum() 
        
    for i in range(1,len(size)):
        pairs_seg = np.asarray(np.where(label == i)).T
        seg_coordinates_dict[i] = pairs_seg
        
    for key,value in seg_coordinates_dict.items():
        x2 = [p[0] for p in value]
        y2 = [p[1] for p in value]
        centroid_seg = (sum(x2) / len(value), sum(y2) / len(value))
        seg_centroid_dict[key] = centroid_seg
    
    dists = []

    for pkey, pvalue in pred_centroid_dict.items():
        
        for skey, svalue in seg_centroid_dict.items():
            
            tmp_dist_list = []
            dist = np.sqrt((pvalue[0] - svalue[0])**2 + (pvalue[1] - svalue[1])**2)
            tmp_dist_list.append(dist)
        
        dists.append(np.asarray(tmp_dist_list).min())
    
    if(len(seg_centroid_dict)!=len(pred_centroid_dict)):
            
        dists.remove(np.asarray(dists).max())
        
    return dists

def naive_count(seg, pred):
    """Scoring method: Return the difference between the number of expected
    clusters and the number of predicted clusters. A positive value indicates
    there were fewer predicted clusters than expected."""

    return count_objects(seg) - count_objects(pred)

def naive_difference(seg, pred):
    """Scoring method: Return the pixel-wise difference between the expected
    image and the predicted image."""

    return count_xor_difference(seg, pred)

def better_count(data):
    return None

def better_difference(data):
    return None

def score(data):
    return {'naive_count':       naive_count(data['seg'], data['masked']),
            'naive_difference':  naive_difference(data['seg'], data['masked']),
            'better_count':      better_count(data),
            'better_difference': better_difference(data)}
