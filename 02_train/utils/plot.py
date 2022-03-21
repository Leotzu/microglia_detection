import matplotlib.pyplot as plt
import numpy as np

import utils.transformations as transform
import utils.metrics as metrics

# helper function to display images
def imshow(data, file=None):
    # Produce one row of plots for each of raw and segmentation, and two
    # plots for prediction: with and without a threshold applied
    rows = len(data)
    cols = data['raw'].shape[0] if data['raw'].ndim > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)

    for row, axis in enumerate(data):
        arr = np.array(data[axis])

        # Ensure all array shapes are in the form (plot, channel, X, Y)
        if arr.ndim == 3:
            arr = arr[np.newaxis]

        for i, plot in enumerate(arr):
            if plot.ndim == 3: # has RGB channels
                plot = plot.transpose(1,2,0) # must be raw array; don't normalize
            else:
                plot = transform.normalize(plot) # must be seg/pred; ensure max = 1, min = 0

            axes[row][i].imshow(plot)

    # Add score labels to the bottom of each column
    for i in range(cols):
        nc = metrics.naive_count(data['seg'][i], data['masked'][i])
        nd = metrics.naive_difference(data['seg'][i], data['masked'][i])
        axes[rows-1][i].set_xlabel(f'{nc}/{nd}')

    if file is None:
        plt.show()
    else:
        plt.savefig(file)

    # Explicitly close the figure to free the memory
    plt.close(fig)
