import matplotlib.pyplot as plt
import numpy as np

# helper function to display images
def imshow(raw, segmentation=None, prediction=None, file=None):
    def norm(arr):
        """Normalize an array to int8 (0-255)"""

        if arr.max() == 0:
            return arr

        arr = arr - arr.min()
        return arr * 255 / arr.max()

    def apply_threshold(arr, threshold):
        newArr = arr.copy()
        newArr[newArr < threshold] = 0
        return newArr

    # Produce one row of plots for each of raw and segmentation, and two
    # plots for prediction: with and without a threshold applied
    rows = len([x for x in (raw, segmentation, prediction, prediction) if x is not None])
    cols = raw.shape[0] if raw.ndim > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)

    for row, arr in enumerate([raw,
                               segmentation,
                               prediction,
                               apply_threshold(prediction, 0.7)]):
        if arr is None:
            continue

        # Ensure all array shapes are in the form (plot, channel, X, Y)
        if arr.ndim == 3:
            arr = arr[np.newaxis]

        for i, plot in enumerate(arr):
            if plot.ndim == 3: # has RGB channels
                plot = plot.transpose(1,2,0) # must be raw array; don't normalize
            else:
                plot = norm(plot) # must be seg/pred; ensure max = 1, min = 0

            axes[row][i].imshow(plot)

    if file is None:
        plt.show()
    else:
        plt.savefig(file)

    # Explicitly close the figure to free the memory
    plt.close(fig)
