import os
import sys
import numpy as np
import zarr
import torch

import utils.argparse as argparse
import utils.plot as plot
import utils.transformations as transform
from utils.metrics import score

def updateZarr(sample, zarr_root):
    if len(sample) != 5:
        print("Unrecognized data format.")
        sys.exit(1)

    zarr_root['raw']   = np.array(sample[:3])
    zarr_root['dots']  = np.array(sample[3])[np.newaxis].astype(np.float32)
    zarr_root['cells'] = np.array(sample[4])[np.newaxis].astype(np.float32)

def runModel(sample, root, model, filename):
    updateZarr(sample, root)
    results = model()
    results['pred'] = transform.normalize(results['pred'])
    results['masked'] = transform.apply_pixel_mask(results['pred'])
    plot.imshow(results, file=filename)

    return results

def logger(suppress=False):
    """Return a lambda to handle output. If the '-d' command-line option is used,
    normal output will be suppressed in favour of only outputting scoring results."""

    def doPrint(s, allow=True):
        if allow:
            print(''.join(s))

    if suppress:
        return lambda *s, allow=False : doPrint(s, allow)
    else:
        return lambda *s, allow=True : doPrint(s) # allow is ignored

def main():
    opts = argparse.parse_args()
    log = logger(opts.print_scores_only)

    # Model weights were not specified on command line args.
    if opts.weights is None:
        basename, ext = os.path.splitext(opts.model_path)
        setattr(opts, 'weights_path', f'{basename}_weights{ext}')

    # Model weights file exists, initialize the model with the edge weights.
    if os.path.exists(opts.weights_path):
        opts.weights = torch.load(opts.weights_path)
        opts.model.load_state_dict(opts.weights)
        opts.model.eval() # docs recommend doing this when loading weights

    log(f'Using model {opts.model_path} and weights {opts.weights_path}')

    # Create/load a sample zarr container for training
    sample_zarr = 'sample.zarr'
    root = zarr.group(store = zarr.DirectoryStore(sample_zarr))
    for lyr in ['raw','dots','cells']:
        if lyr not in root:
            root.array(str(lyr),np.array(0)) # create a placeholder

    #scores = []
    sample_num = 0
    if opts.type in ['train','both']:
        # For each image, populate the sample zarr container and train on it
        for i, sample in enumerate(opts.data['train']):
            sampleName = opts.data['train'].attrs['name'][i]

            log(f"Training on sample {i+1} of {len(opts.data['train'])}",
                    f"({sampleName})")
            results = runModel(sample, root,
                lambda: opts.train(opts.model, sample_zarr),
                f"train-{sampleName}")

            torch.save(opts.model, opts.model_path)
            torch.save(opts.model.state_dict(), opts.weights_path)

            # Score results and print
            sample_score = score(results, sample_num, sampleName)
            log('\n'.join([str(s) for s in sample_score]), allow=True)
            sample_num += len(sample_score)

    if opts.type in ['test','both']:
        for i, sample in enumerate(opts.data['test']):
            sampleName = opts.data['test'].attrs['name'][i]

            log(f"Testing on sample {i+1} of {len(opts.data['test'])}",
                    f"({sampleName})")
            results = runModel(sample, root,
                lambda: opts.test(opts.model, sample_zarr),
                f"test-{sampleName}")

            # Score results and print
            sample_score = score(results, sample_num, sampleName)
            log('\n'.join([str(s) for s in sample_score]), allow=True)
            sample_num += len(sample_score)

if __name__ == '__main__':
    main()
