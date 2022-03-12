import os
import sys
import numpy as np
import zarr
import torch

import utils.argparse as argparse
import utils.plot as plot

def main():
    opts = argparse.parse_args()

    # Model weights were not specified on command line args.
    if opts.weights is None:
        basename, ext = os.path.splitext(opts.model_path)
        setattr(opts, 'weights_path', f'{basename}_weights{ext}')

    # Model weights file exists, initialize the model with the edge weights.
    if os.path.exists(opts.weights_path):
        opts.weights = torch.load(opts.weights_path)
        opts.model.load_state_dict(opts.weights)
        opts.model.eval() # docs recommend doing this when loading weights

    print(f'Using model {opts.model_path} and weights {opts.weights_path}')

    # Create/load a sample zarr container for training
    sample_zarr = 'sample.zarr'
    root = zarr.group(store = zarr.DirectoryStore(sample_zarr))
    for lyr in ['raw','dots','cells']:
        if lyr not in root:
            root.array(str(lyr),np.array(0)) # create a placeholder

    # For each image, populate the sample zarr container and train on it
    for i, image in enumerate(opts.data['train']):
        print(f'Training on image {i+1} of {len(opts.data["train"])}')
        if len(image) != 5:
            print("Unrecognized data format.")
            sys.exit(1)

        root['raw']   = np.array(image[:3])
        root['dots']  = np.array(image[3])[np.newaxis].astype(np.float32)
        root['cells'] = np.array(image[4])[np.newaxis].astype(np.float32)

        results = opts.train(opts.model, sample_zarr)
        plot.imshow(results['raw'], results['seg'], results['prediction'], file=f'train-{i}.png')

    torch.save(opts.model, opts.model_path)
    torch.save(opts.model.state_dict(), opts.weights_path)

if __name__ == '__main__':
    main()
