from scipy import ndimage
import numpy as np

# Counts the number of objects (clumps) in a 2d numpy array
# pixel threshold is pixel intensity to consider as an object
# size threshold is the size of object to consider an object (ie 1 pixel might be too small to consider)
def count_objects(data, pixel_threshold=0.7, size_threshold=1):
    
    data[data > pixel_threshold] = 1
    data[data <= pixel_threshold] = 0
    
    label, _ = ndimage.label(data == 1)
    size = np.bincount(label.ravel())

    # ignore the 0'th position, which will be the unactivated pixels
    num_objects = (size[1:] > size_threshold).sum()
    
    return num_objects

def count_xor_difference(arr1, arr2, threshold=0.7):
    """Count the number of pixels that differ between two numpy arrays.

    arr1,arr2 -- arrays to compare
    threshold -- array elements above this threshold will be considered
                 'on', and 'off' if below or equal to.
    """

    return np.count_nonzero(np.logical_xor(arr1 > threshold,
                                           arr2 > threshold))

def get_centroids(arr, pixel_threshold=0.7, size_threshold=2):
    """Return x,y coordinates of centroids of clusters in image."""

    label, num_label = ndimage.label(arr > pixel_threshold)
    size = np.bincount(label.ravel())

    centroids = []
    for i in range(1, len(size)):
        if size[i] < size_threshold:
            continue

        x,y = np.array(label == i).nonzero()
        centroids.append((x.mean(), y.mean()))

    return np.array(centroids)

def get_centroid_distance_matrix(seg, pred):
    """Compute a matrix containing the distances between all pairs of centroids
    in the seg and pred arrays."""

    seg_centroids = get_centroids(seg)
    pred_centroids = get_centroids(pred)

    dist = np.zeros((len(seg_centroids), len(pred_centroids)))
    if 0 in dist.shape: # one of seg or pred have zero centroids
        return None

    for i in range(len(dist)):
        dist[i] = np.linalg.norm(seg_centroids[i] - pred_centroids, axis=1)

    return dist

# Computes the distances between the centroids of the predicted
# objects to the centroids of the actual objects
# Returns a list of distances
def compute_centroids(seg, pred, pixel_threshold=0.7, size_threshold=1):
    
    seg_coordinates_dict = {}
    pred_coordinates_dict = {}
    
    seg_centroid_dict = get_centroids(seg, pixel_threshold, size_threshold)
    pred_centroid_dict = get_centroids(pred, pixel_threshold, size_threshold)

    s = np.stack([seg_centroid_dict[x] for x in seg_centroid_dict.keys()])
    p = np.stack([pred_centroid_dict[x] for x in pred_centroid_dict.keys()])

    d = np.zeros((len(s),len(p)))
    for i in range(len(s)):
        d[i] = np.linalg.norm(s[i] - p, axis=1)
    m = np.argmin(d, axis=1)

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
    clusters and the number of predicted clusters."""

    return abs(count_objects(seg) - count_objects(pred))

def naive_difference(seg, pred):
    """Scoring method: Return the pixel-wise difference between the expected
    image and the predicted image."""

    return count_xor_difference(seg, pred)

def better_count(seg, pred, tolerance=5):
    """Scoring method: Return the number of clusters that do not match between
    the expected and predicted results."""

    score = 0
    for s, p in zip(seg, pred):
        s_centroids = get_centroids(np.squeeze(s))
        p_centroids = get_centroids(np.squeeze(p))

        if len(s_centroids) > 0 and len(p_centroids) > 0:
            for c in s_centroids.copy():
                match = np.linalg.norm(c - p_centroids, axis=1) < tolerance

                if match.any():
                    # Found a matching centroid between seg and pred, so
                    # remove it from both arrays so that we don't count it
                    # twice
                    p_centroids = p_centroids[np.logical_not(match)]
                    s_centroids = s_centroids[np.all(s_centroids != c, axis=1)]

        # Remaining centroids do not agree, so score is their sum
        score += len(s_centroids) + len(p_centroids)

    return score

def better_difference(data):
    """Scoring method: for each centroid, compute its cluster's pixel-wise
    difference with the corresponding cluster in the other layer. Clusters
    that do not match the other layer will have all pixels count towards the
    error."""

    # Not yet implemented
    return None

def centroid_deviation(seg, pred):
    """Scoring method: Return the sum of min deviation between the expected
    centroids and the predicted."""

    score = 0
    for s, p in zip(seg, pred):
        dist = get_centroid_distance_matrix(np.squeeze(s), np.squeeze(p))
        if dist is not None:
            score += np.amin(dist, axis=1).sum()

    return score

def score(data):
    return {'naive_count':        naive_count(data['seg'], data['masked']),
            'naive_difference':   naive_difference(data['seg'], data['masked']),
            'better_count':       better_count(data['seg'], data['masked']),
            'better_difference':  better_difference(data),
            'centroid_deviation': centroid_deviation(data['seg'], data['masked'])}
