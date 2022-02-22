import matplotlib.pyplot as plt
import numpy as np
import random
import zarr
from PIL import Image

# convert raw and groundtruth images into np.arrays
raw_raster = Image.open('008_x3514_y764_600x600_raw.tif')
gt_raster = Image.open('008_x3514_y764_600x600_seg_dia10.tif')

# transpose data so it's in the form: (channels, x, y)
raw_data = np.array(raw_raster).transpose(2,0,1) 

gt_data = np.array(gt_raster)
gt_data = gt_data[np.newaxis,:].astype(np.float32) #Q: why float32??

# store the images in a zarr container
f = zarr.open('sample_data.zarr', 'w')
f['raw'] = raw_data
f['raw'].attrs['resolution'] = (1, 1)
f['ground_truth'] = gt_data
f['ground_truth'].attrs['resolution'] = (1, 1)

# helper function to show image(s)
def imshow(raw, ground_truth=None, prediction=None):
	rows = 1
	if ground_truth is not None:
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
	if ground_truth is not None:
		if len(ground_truth.shape) == 3:
			axes[row][0].imshow(ground_truth[0])
		else:
			for i, gt in enumerate(ground_truth):
				axes[row][i].imshow(gt[0])
		row += 1
	if prediction is not None:
		if len(prediction.shape) == 3:
			axes[row][0].imshow(prediction[0])
		else:
			for i, gt in enumerate(prediction):
				axes[row][i].imshow(gt[0])
	plt.show()

# Call imshow() to display the raw and gt images we're working with
imshow(zarr.open('sample_data.zarr')['raw'][:])
imshow(zarr.open('sample_data.zarr')['ground_truth'][:])


# ===============================================
# Now onto the actual gunpowder pipeline building
# ===============================================
import gunpowder as gp

# declare arrays to use in the pipeline
raw = gp.ArrayKey('Raw')

# Create the "pipeline", consisting only of a data source
source = gp.ZarrSource(
	'sample_data.zarr', # the zarr container
	{raw: 'raw'}, # which dataset to associate to the array key
	{raw: gp.ArraySpec(interpolatable=True)} # meta-information
)
pipeline = source

# Forumalte a request for "raw" 
# 	-> i.e. perform a "BatchRequest" for raw date (in this case, starting at (0,0) with a size of (64,128)
# 	-> this request acts like a dictionary, mapping each array key to the ROI (region of interest)
request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (64, 128))

# Now build the pipeline... *************** Important
with gp.build(pipeline):
	#... and request the batch ****************** Important
	batch = pipeline.request_batch(request)

# show the content of the batch
print(f"batch returned: {batch}")
#imshow(batch[raw].data)

# You can also see what the request looks like by printing it
print(request)

# ===============================================
# *** RANDOM SAMPLES ***
# 	-> in training pipelines, it's often useful to randomly select a location to crop data from. GP provides a node "RandomLocation" that does this automatically

# add a RandomLocation to the pipeline to randomly select a sample
random_location = gp.RandomLocation()
pipeline = source + random_location

# Printing the pipeline will show that it's using a RandomLocation
#print(pipeline)

# Now we can issue the same type of request as before, except we're now using random_location:
with gp.build(pipeline):
	batch = pipeline.request_batch(request)
#imshow(batch[raw].data)

# ===============================================
# *** Simple Augmentation ***
simple_augment = gp.SimpleAugment()
pipeline = source + random_location + simple_augment

with gp.build(pipeline):
	batch = pipeline.request_batch(request)
#imshow(batch[raw].data)

# ===============================================
# *** ElasticAugment ***
#	-> allows you to elastically deform a batch
import math

elastic_augment = gp.ElasticAugment(
	control_point_spacing = (16, 16),
	jitter_sigma = (4.0, 4.0),
	rotation_interval = (0, math.pi/2)
	)
# redefine the pipeline with the added augmentation
pipeline = source + random_location + simple_augment + elastic_augment

# build the pipeline and request the batch (as we've done twice now)
with gp.build(pipeline):
	batch = pipeline.request_batch(request)

# finally print the augmented batch
#imshow(batch[raw].data)

# ===============================================
# *** INTENSITY AUGMENTATION ***
# 	-> intensity values can be modified, and random noise can be added

normalize = gp.Normalize(raw)
intensity_augment = gp.IntensityAugment(
	raw,
	scale_min = 0.8,
	scale_max = 1.2,
	shift_min = -0.2,
	shift_max = 0.2)
# create some noise now
noise_augment = gp.NoiseAugment(raw)

# Now again we redefine the pipeline, buil it, then request the batch:
pipeline = (
	source + 
	normalize + 
	random_location +
	simple_augment +
	elastic_augment +
	intensity_augment +
	noise_augment)
with gp.build(pipeline):
	batch = pipeline.request_batch(request)
#imshow(batch[raw].data)

# ===============================================
# *** STACK node ***
# 	-> this allows us to create a batch with multiple samples drawn from the same pipeline definition (but will be different if random_location is called, as it is in the following example)

stack = gp.Stack(5)
pipeline = (
	source + 
	normalize + 
	random_location +
	simple_augment +
	elastic_augment +
	intensity_augment +
	noise_augment +
	stack)

with gp.build(pipeline):
	batch = pipeline.request_batch(request)
#imshow(batch[raw].data)

# ===============================================
# *** Requesting Multiple Arrays ***

# Call imshow() to display the raw image and the "ground_truth" that we're working with
'''
imshow(
    zarr.open('sample_data.zarr')['raw'][:],
    zarr.open('sample_data.zarr')['ground_truth'][:]
    )
'''
# modify our source node and our request so we can request raw gata and segmentation at the same time:

gt = gp.ArrayKey('GROUND_TRUTH')
source = gp.ZarrSource(
    'sample_data.zarr',
    {
        raw: 'raw',
        gt: 'ground_truth'
    },
    {
        raw: gp.ArraySpec(interpolatable=True),
        gt: gp.ArraySpec(interpolatable=False)
    })

request[gt] = gp.Roi((0, 0), (64, 128))

pipeline = (
    source +
    normalize +
    random_location +
    simple_augment +
    elastic_augment +
    intensity_augment +
    noise_augment +
    stack)
    
with gp.build(pipeline):
    batch = pipeline.request_batch(request)
    
#imshow(batch[raw].data, batch[gt].data)

# We can even offset the ground_truth ROI from the raw ROI by "requesting an offset ROI for the ground-truth":
#   -> Note: this means that GP requests can contain ROIs with different offsets and sizes.
odd_request = request.copy()
odd_request[gt] = gp.Roi((20,20),(64,128))

with gp.build(pipeline):
    batch = pipeline.request_batch(odd_request)

#imshow(batch[raw].data, batch[gt].data)

# ===============================================
# *** Multiple Sources (i.e. Zarr containers) ***
#   -> building from the previous example, GP allows you to have multiple sources for different arrays and merge them together into one:

# source for raw array
source_raw = gp.ZarrSource(
    'sample_data.zarr',
    {raw: 'raw'},
    {raw: gp.ArraySpec(interpolatable=True)}
)
# source for ground-truth array
source_gt = gp.ZarrSource(
    'sample_data.zarr',
    {gt: 'ground_truth'},
    {gt: gp.ArraySpec(interpolatable=False)}
)
# now we can merge these two sources to assemble the pipeline:
combined_source = (source_raw, source_gt) + gp.MergeProvider()

pipeline = (
    combined_source +
    normalize +
    random_location +
    simple_augment +
    elastic_augment +
    intensity_augment +
    noise_augment +
    stack)

with gp.build(pipeline):
    batch = pipeline.request_batch(request)

#imshow(batch[raw].data, batch[gt].data)


# ===============================================
# *** TRAINING A NETWORK ***
# ===============================================
#   -> So far we've only seen how to use GP in order to generate training batches (but not actually train anything).
#   -> Now, using the same pipeline structure, we'll show how we can train a neural network directly using GP.
#   -> we will train a 2D U-net on the binary ground-truth using a binary cross entropy loss (the model and loss links are provided in the tutorial).

# Required: 
#   -> pip install git+https://github.com/funkelab/funlib.learn.torch
#   -> pip install torch (<- this is pytorch)

import torch
from funlib.learn.torch.models import UNet, ConvPass

# to make sure I get the same results as the tutorial:
torch.manual_seed(18)

# Define the UNet
#   -> def: a CNN that was developed for biomedical image segmentation.
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

loss = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())

# *** FEATURE ***
#   -> The UNet training can be implemented in a GP node (in this case, "torch.Train").
#   -> The main benefit of using this node in a GP pipeline (as opposed to taking the batches we already requested and feeding them manually into the model), is that the output of the network itself can be mapped t o a GP array, which can subsequently be used in the pipeline.

# Now we will create a new array key ("prediction") to implement the training in a GP node (as described above):

# create a new array key for the network output
prediction = gp.ArrayKey('PREDICTION')

# create a train node using our model, loss, and optimizer
train = gp.torch.Train(
    model,
    loss,
    optimizer,
    inputs = {
        'input': raw
    },
    loss_inputs = { # *** Q: What are loss inputs?? -> A: in API notes
        0: prediction,
        1: gt
    },
    outputs = {
        0: prediction
    })
    
# Now add "train" to the pipeline
pipeline = (
    source +
    normalize +
    random_location +
    simple_augment +
    elastic_augment +
    intensity_augment +
    noise_augment +
    stack +
    train)
    
# now add the prediction to the request
request[prediction] = gp.Roi((0,0), (64,128))

# Finally we can build and display like normal:
with gp.build(pipeline):
    batch = pipeline.request_batch(request)
# include the prediction when calling imshow()
#imshow(batch[raw].data, batch[gt].data, batch[prediction].data)

# To get better results, we can train the same model for a few more iterations:
with gp.build(pipeline):
    for i in range(10):
        batch = pipeline.request_batch(request)
        
imshow(batch[raw].data, batch[gt].data, batch[prediction].data)

# *** FEATURE ***
#   -> In addition to doing the training in GP, we can also perform the FINAL PREDICTIONS in GP once the training is complete. 

# Starting with the model we previously trained, we now apply it on the entire image.
#   -> We do this by chunking the image into sections of the correct size, then applying the network to each chunk in a scanning fashion.
#   -> We then reassemble the individually chunked sections into the entire image.
#   -> This is all done in another GP node: "Scan" <- *********

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
    
stack = gp.Stack(1) # All we want is one final image of the whole pic

# request matching the model input and output sizes:
scan_request = gp.BatchRequest()
scan_request[raw] = gp.Roi((0,0), (64,128))
scan_request[prediction] = gp.Roi((0,0), (64,128))

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


# ===============================================
# *** Prediction in Large nD Arrays ***
# ===============================================
# Problem: 
#   -> The above code involves us using gp.Scan() for the whole image, but what if the arrays are too big to fit into memory?
# Solution:
#   -> GP was made to work with arbitrarily large nD arrays (meaning ... ??)
#   -> This means (I think...) that we can break the predictions down into n chucks and store them, for example, in a Zarr container.
#   -> To do this we add a node between predict and scan, through which every batch will pass before it is discarded.
#   -> We will use the ZarrWrite node to do this: assemble a Zarr container of all the arrays passing through it.

# prepare the zarr dataset to write to
f = zarr.open('sample_data.zarr')
ds = f.create_dataset('prediction', shape=(1,1,600,600))
ds.attrs['resolution'] = (1,1)
ds.attrs['offset'] = (0,0)

# create a ZarrWrite() node in order to store the predictions
zarr_write = gp.ZarrWrite(
    output_filename = 'sample_data.zarr',
    dataset_names = {
        prediction: 'prediction'
    })

pipeline = (
    source +
    normalize +
    stack +
    predict +
    zarr_write +
    scan)
    
# request an empty batch from scan
request = gp.BatchRequest()
 
with gp.build(pipeline):
    batch = pipeline.request_batch(request)
    
print(batch)
# call imshow() for the zarr containers that contain the raw image and the predicted image.
imshow(
    zarr.open('sample_data.zarr')['raw'][:],
    None,
    zarr.open('sample_data.zarr')['prediction'][:])


