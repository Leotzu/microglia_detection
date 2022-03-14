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

    rows = sum([1 for x in (raw, segmentation, prediction) if x is not None])
    cols = raw.shape[0] if len(raw.shape) > 3 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4), sharex=True, sharey=True, squeeze=False)

    if len(raw.shape) == 3:
        axes[0][0].imshow(raw.transpose(1, 2, 0))
    else:
        for i, im in enumerate(raw):
            axes[0][i].imshow(im.transpose(1,2,0))

    row = 1
    if segmentation is not None:
        if len(segmentation.shape) == 3:
            axes[row][0].imshow(norm(segmentation[0]))
        else:
            for i, seg in enumerate(segmentation):
                axes[row][i].imshow(norm(seg[0]))
        row += 1

    if prediction is not None:
        if len(prediction.shape) == 3:
            axes[row][0].imshow(norm(prediction[0]))
        else:
            for i, seg in enumerate(prediction):
                axes[row][i].imshow(norm(seg[0]))

    if file is None:
        plt.show()
    else:
        plt.savefig(file)

    # Explicitly close the figure to free the memory
    plt.close(fig)
