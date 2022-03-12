import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import zarr
import gunpowder as gp
import math
import torch
from funlib.learn.torch.models import UNet, ConvPass

# helper function to display images
def imshow(raw, segmentation=None, prediction=None, file=None):
    def norm(arr):
        """Normalize an array to int8 (0-255)"""

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

# =======================================
# Build the gunpowder pipeline
# =======================================

#TODO: split out validation routine
#TODO: it might not make sense to save an image from here, since we're going
#       to train on multiple images. Either save with a unique name, or output
#       a metric to stdout.
def train(model, zarrfile):
    # first we declare arrays to use in the pipeline and to later prediction results
    raw = gp.ArrayKey('raw')
    seg = gp.ArrayKey('cells')
    prediction = gp.ArrayKey('PREDICTION')

    # Create a Zarr Source node that houses the zarr file containing our input images
    source = gp.ZarrSource(
        zarrfile, # the zarr container
        {
            raw: 'raw', # which dataset to associate to the array key
            seg: 'cells'
        },
        {
            raw: gp.ArraySpec(interpolatable=True,
                    voxel_size=(1,1)),
            seg: gp.ArraySpec(interpolatable=False,
                    voxel_size=(1,1)),
        })

    # Create a Simple Augmentation node (mirrors and/or transposes image)
    simple_augment = gp.SimpleAugment()

    # Create an Elastic Augmentation node (elastically deforms the image)
    #elastic_augment = gp.ElasticAugment(
    #    control_point_spacing = (16, 16),
    #    jitter_sigma = (4.0, 4.0),
    #    rotation_interval = (0, math.pi/2)
    #    )
        
    # add a RandomLocation to the pipeline to randomly select samples for a stack
    random_location = gp.RandomLocation()

    # Create a Stack node to create a batch with multiple samples drawn from the same pipline definition
    stack = gp.Stack(10)

    # Create a Normalize node to ensure we're dealing with floats between 0 and 1
    normalize = gp.Normalize(raw)

    # Define the dimensions of your batch squares (described below)
    square_len = 200

    # Create a batch request, which acts like a dictionary mapping each array key
    # to the ROI (region of interest)
    # We want to be able to request the images of the raw, segmented and/or prediction data:
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0, 0), (square_len, square_len))
    request[seg] = gp.Roi((0, 0), (square_len, square_len))
    request[prediction] = gp.Roi((0,0), (square_len,square_len))

    # ===============================================
    # Train the neural network
    # ===============================================

    # Define the loss to be binary cross entropy
    model.train()
    loss = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters())

    # create a GP train node using our model, loss, and optimizer
    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs = {
            'input': raw
        },
        loss_inputs = {
            0: prediction,
            1: seg
        },
        outputs = {
            0: prediction
        })
        
    # Now we construct the pipeline by concatinating all of our GP nodes
    pipeline = (
        source +
        normalize +
        random_location +
        simple_augment +
        #elastic_augment +
        stack +
        train)

    # Finally we can build the pipeline and display it
    # We can train the same model for a few iterations to get better results
    with gp.build(pipeline):
        for i in range(100):
            batch = pipeline.request_batch(request)
            
    imshow(batch[raw].data, batch[seg].data, batch[prediction].data, file='train.png')

    # ===============================================
    # Evaluate/Test the network
    # ===============================================

    # set model into evaluation mode
    model.eval()

    # Here we replace gp.torch.Train() (what we had before) with it's equivalent, gp.torch.Predict().
    predict = gp.torch.Predict(
        model,
        inputs = {
            'input': raw
        },
        outputs = {
            0: prediction
        })
        
    # All we want is one output image of our entire test image
    stack = gp.Stack(1)

    # request matching the model input and output sizes:
    scan_request = gp.BatchRequest()
    scan_request[raw] = gp.Roi((0,0), (square_len,square_len))
    scan_request[prediction] = gp.Roi((0,0), (square_len,square_len))

    # create the gp.Scan node
    scan = gp.Scan(scan_request)

    pipeline = (
        source +
        normalize +
        stack +
        predict +
        scan)

    # request for raw and prediction for the whole image
    request = gp.BatchRequest()
    request[raw] = gp.Roi((0,0), (500,500)) # Size of the whole image
    request[prediction] = gp.Roi((0,0), (500,500))

    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    imshow(batch[raw].data,
        np.array(zarr.open(zarrfile)[seg])[np.newaxis],
        batch[prediction].data,
        file='final.png')

def initModel():
    # Define the UNet
    unet = UNet(
        in_channels=3,
        num_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=[[2,2], [2,2]],
        kernel_size_down=[[[3,3], [3,3]]]*3,
        kernel_size_up=[[[3,3], [3,3]]]*2,
        padding='same')
        
    # Define the model
    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=4,
                 out_channels=1,
                 kernel_sizes=[(1,1)], # 2 dimensions => Conv2d
                 activation=None),
        torch.nn.Sigmoid())
    
    return model

class LoadZarrAction(argparse.Action):
    """Load and validate zarr data from command line arg."""

    def __call__(self, parser, namespace, zarr_dir, option_string=None):
        try:
            if os.path.exists(zarr_dir) and \
                    os.path.isdir(zarr_dir) and \
                    zarr_dir.strip('\/')[-5:] == '.zarr':

                z = zarr.open(zarr_dir)
                for lyr in ['train', 'test']:
                    if lyr not in z:
                        raise Exception # invalid zarr

                setattr(namespace, 'data', z)
                setattr(namespace, 'zarr', zarr_dir)

            else:
                raise Exception
        except zarr.errors.FSPathExistNotDir:
            parser.error(f'Invalid zarr: {zarr_dir}')
        except:
            parser.error(f'Invalid zarr or not found: {zarr_dir}')

class LoadModelAction(argparse.Action):
    """Load and validate PyTorch model from command line arg."""

    def __call__(self, parser, namespace, model_path, option_string=None):
        try:
            if os.path.exists(model_path):
                model = torch.load(model_path)
            else:
                model = initModel()
            setattr(namespace, 'model', model)
            setattr(namespace, 'model_path', model_path)
        except FileNotFoundError:
            parser.error(f'File not found: {model_path}')
        except:
            parser.error(f'Invalid PyTorch model: {model_path}')

class LoadModelWeightsAction(argparse.Action):
    """Load and validate PyTorch model weights from command line arg."""

    def __call__(self, parser, namespace, weights_path, option_string=None):
        try:
            weights = torch.load(weights_path)
            setattr(namespace, 'weights', weights)
            setattr(namespace, 'weights_path', weights_path)
        except FileNotFoundError:
            parser.error(f'File not found: {weights_path}')
        except:
            parser.error(f'Invalid PyTorch model: {weights_path}')

def parse_args():
    """Interpet command line args. Returns opts dictionary."""

    parser = argparse.ArgumentParser(
        description='A Convolutional Neural Network to identify microglia in mouse brain cell samples.')
    parser.add_argument('zarr', metavar='zarr',
        action=LoadZarrAction,
        help='Zarr directory containing input data and segmentation layers.')
    parser.add_argument('model', metavar='model.pth', default='model.pth',
        action=LoadModelAction,
        help="Specify a .pth file to use as the initial model. If it doesn't exist, it will be initialized.")
    parser.add_argument('-w', '--weights', metavar='pth',
        action=LoadModelWeightsAction,
        help='Optional: specify a .pth file to use as the initial model weights.')

    return parser.parse_args()

def main():
    opts = parse_args()

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

        train(opts.model, sample_zarr)

    torch.save(opts.model, opts.model_path)
    torch.save(opts.model.state_dict(), opts.weights_path)

if __name__ == '__main__':
    main()
