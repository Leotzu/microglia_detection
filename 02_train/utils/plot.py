import matplotlib.pyplot as plt
import numpy as np

import utils.metrics as metrics

# helper function to display images
def imshow(data, file=None):
    # Produce one row of plots for each of raw and segmentation, and two
    # plots for prediction: with and without a threshold applied
    rows = len(data)
    cols = data['raw'].shape[0] if data['raw'].ndim > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)

    if rows > cols: # convert figsize from landscape to portrait
        fig.set_size_inches(np.flip(fig.get_size_inches()))

    for row, axis in enumerate(data):
        arr = np.array(data[axis])

        # Ensure all array shapes are in the form (plot, channel, X, Y)
        if arr.ndim == 3:
            arr = arr[np.newaxis]

        for i, plot in enumerate(arr):
            if plot.ndim == 3: # has RGB channels
                plot = plot.transpose(1,2,0) # must be raw array; don't normalize

            axes[row][i].imshow(plot)

    # Add score labels to the bottom of each column
    for i in range(cols):
        nc = metrics.naive_count(data['seg'][i], data['masked'][i])[0]
        bc = metrics.better_count(data['seg'][i], data['masked'][i])[0]
        nd = metrics.naive_difference(data['seg'][i], data['masked'][i])[0] / data['masked'][i].size * 100
        axes[rows-1][i].set_xlabel(f'nc:{nc}\nbc:{bc}\nnd:{nd:.2f}')

    fig.set_tight_layout(True)

    if file is None:
        plt.show()
    else:
        plt.savefig(file)

    # Explicitly close the figure to free the memory
    plt.close(fig)
