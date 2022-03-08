import matplotlib.pyplot as plt
import numpy as np
import zarr
import gunpowder as gp
import math
import torch
from funlib.learn.torch.models import UNet, ConvPass

# ========================================
# Initial setup and image display function
# ========================================

# Change this to your target input.zarr location (i.e. the container your data is stored in)
data_dir = '008/trainingset_01'

# This is where the INPUT raw and segmented images are stored
zarr_dir = f'../../01_data/{data_dir}/input.zarr'
# This is where the OUTPUT predictions will be stored after training and testing
output_zarr_dir = f'results/{data_dir}/prediction.zarr'

# helper function to display images
def imshow(raw, segmentation=None, prediction=None):
	rows = 1
	if segmentation is not None:
		rows += 1
	if prediction is not None:
		rows += 1
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
			axes[row][0].imshow(segmentation[0])
		else:
			for i, seg in enumerate(segmentation):
				axes[row][i].imshow(seg[0])
		row += 1
	if prediction is not None:
		if len(prediction.shape) == 3:
			axes[row][0].imshow(prediction[0])
		else:
			for i, seg in enumerate(prediction):
				axes[row][i].imshow(seg[0])
	plt.show()

# Call imshow() to display the raw and segmentation images we're working with
imshow(zarr.open(zarr_dir)['raw'][:], zarr.open(zarr_dir)['seg'][:])

# =======================================
# Build the gunpowder pipeline
# =======================================

# first we declare arrays to use in the pipeline and to later prediction results
raw = gp.ArrayKey('Raw')
seg = gp.ArrayKey('Seg')
prediction = gp.ArrayKey('PREDICTION')

# Create a Zarr Source node that houses the zarr file containing our input images
source = gp.ZarrSource(
    zarr_dir, # the zarr container
    {
        raw: 'raw', # which dataset to associate to the array key
        seg: 'seg'
    },
    {
        raw: gp.ArraySpec(interpolatable=True),
       seg: gp.ArraySpec(interpolatable=False)
    })

# Create a Simple Augmentation node (mirrors and/or transposes image)
simple_augment = gp.SimpleAugment()

# Create an Elastic Augmentation node (elastically deforms the image)
elastic_augment = gp.ElasticAugment(
	control_point_spacing = (16, 16),
	jitter_sigma = (4.0, 4.0),
	rotation_interval = (0, math.pi/2)
	)
	
# add a RandomLocation to the pipeline to randomly select samples for a stack
random_location = gp.RandomLocation()

# Create a Stack node to create a batch with multiple samples drawn from the same pipline definition
stack = gp.Stack(10)

# Create a Normalize node to ensure we're dealing with floats between 0 and 1
normalize = gp.Normalize(raw)

# Define the dimensions of your batch squares (described below)
square_len = 200

# Create a batch request, which acts like a dictionary mapping each array key to the ROI (region of interest)
# We want to be able to request the images of the raw, segmented and/or prediction data:
request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (square_len, square_len))
request[seg] = gp.Roi((0, 0), (square_len, square_len))
request[prediction] = gp.Roi((0,0), (square_len,square_len))

# ===============================================
# Train the neural network
# ===============================================

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
    ConvPass(4, 1, [(1,1)], activation=None),
    torch.nn.Sigmoid())

# Define the loss to be binary cross entropy
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
    elastic_augment +
    stack +
    train)

# Finally we can build the pipeline and display it
# We can train the same model for a few iterations to get better results
with gp.build(pipeline):
    for i in range(100):
        batch = pipeline.request_batch(request)
        
imshow(batch[raw].data, batch[seg].data, batch[prediction].data)

# ===============================================
# Evaluate/Test the network
# ===============================================

# set model into evaltuation mode
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
request[raw] = gp.Roi((0,0), (600,600)) # Size of the whole image
request[prediction] = gp.Roi((0,0), (600,600))

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

imshow(batch[raw].data, None, batch[prediction].data)

