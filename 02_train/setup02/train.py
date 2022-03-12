import numpy as np
import zarr
import gunpowder as gp
import math
import torch

def array_keys():
    raw = gp.ArrayKey('raw')
    seg = gp.ArrayKey('cells')
    prediction = gp.ArrayKey('PREDICTION')

    return raw, seg, prediction
    
def common(model, zarrfile):
    raw, seg, prediction = array_keys()

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

    # Create a Normalize node to ensure we're dealing with floats between 0 and 1
    normalize = gp.Normalize(raw)

    return source, normalize

def train(model, zarrfile):
    # first we declare arrays to use in the pipeline and to later prediction results
    raw, seg, prediction = array_keys()
    source, normalize = common(model, zarrfile)

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
            
    return {'raw': batch[raw].data, 
            'seg': batch[seg].data,
            'prediction': batch[prediction].data}

def test(model, zarrfile):
    # ===============================================
    # Evaluate/Test the network
    # ===============================================

    raw, seg, prediction = array_keys()

    # set model into evaluation mode
    model.eval()

    source, normalize = common(model, zarrfile)

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

    return {'raw': batch[raw].data,
            'seg': np.array(zarr.open(zarrfile)[seg])[np.newaxis],
            'prediction': batch[prediction].data}
