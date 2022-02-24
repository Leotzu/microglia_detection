import matplotlib.pyplot as plt
import numpy as np
import zarr
import gunpowder as gp
import math
import torch
from funlib.learn.torch.models import UNet, ConvPass

# =======================================
# Initial setup:
# =======================================

# Change this to your target zarr.zarr location (i.e. the container your data is stored in)
data_dir = '008/trainingset_01'

# This is where the INPUT raw and segmented images are stored
# (as dictated by 01_data/create_zarr.py)
zarr_dir = f'../../01_data/{data_dir}/zarr.zarr'
# This is where the OUTPUT predictions will be stored after 
#we send in the entire raw image as our first test of the model
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
imshow(zarr.open(zarr_dir)['raw'][:])
imshow(zarr.open(zarr_dir)['seg'][:])


# =======================================
# Now let's build the gunpowder pipeline:
# =======================================

# A GP pipeline is a concatination of a bunch of "nodes" that primarily contain data, data processing commands, and training parameters. 
# You therefore define your nodes, concatinate them all together in the pipeline, "build" the pipeline (which will train if you have a training node in the pipeline), then request information from the built pipeline (like predictions, for example).

# first we declare arrays to use in the pipeline
raw = gp.ArrayKey('Raw')
seg = gp.ArrayKey('Seg')

# Create a data source node that contains the zarr file that has our images stored inside.
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
    
# ==============================================================
# ==============================================================
# ==============================================================
# THIS IS MY EDITING BOOKMARK
# ==============================================================
# ==============================================================
# ==============================================================


request = gp.BatchRequest()
request[raw] = gp.Roi((0, 0), (64, 128))
request[seg] = gp.Roi((0, 0), (64, 128))
'''

# add a RandomLocation to the pipeline to randomly select a sample
'''
random_location = gp.RandomLocation()
'''

# ===============================================
# *** Simple Augmentation ***
'''
simple_augment = gp.SimpleAugment()
'''

# ===============================================
# *** ElasticAugment ***
#	-> allows you to elastically deform a batch 
'''
elastic_augment = gp.ElasticAugment(
	control_point_spacing = (16, 16),
	jitter_sigma = (4.0, 4.0),
	rotation_interval = (0, math.pi/2)
	)
'''

# ===============================================
# *** INTENSITY AUGMENTATION ***
# 	-> intensity values can be modified, and random noise can be added
'''
normalize = gp.Normalize(raw)
intensity_augment = gp.IntensityAugment(
	raw,
	scale_min = 0.8,
	scale_max = 1.2,
	shift_min = -0.2,
	shift_max = 0.2)

# create some noise now
noise_augment = gp.NoiseAugment(raw)
'''

# ===============================================
# *** STACK node ***
# 	-> this allows us to create a batch with multiple samples drawn from the same pipeline definition (but will be different if random_location is called, as it is in the following example)
'''
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
    
imshow(batch[raw].data, batch[seg].data)


# ===============================================
# *** TRAINING A NETWORK ***
# ===============================================
#   -> So far we've only seen how to use GP in order to generate training batches (but not actually train anything).
#   -> Now, using the same pipeline structure, we'll show how we can train a neural network directly using GP.
#   -> we will train a 2D U-net on the binary segmentation using a binary cross entropy loss (the model and loss links are provided in the tutorial).

# to make sure I get the same results as the tutorial:
torch.manual_seed(18)

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

loss = torch.nn.BCELoss()

optimizer = torch.optim.Adam(model.parameters())

# *** FEATURE ***
#   -> The UNet training can be implemented in a GP node (in this case, "torch.Train").
#   -> The main benefit of using this node in a GP pipeline (as opposed to taking the batches we already requested and feeding them manually into the model), is that the output of the network itself can be mapped to a GP array, which can subsequently be used in the pipeline.

# Now we will create a new array key ("prediction") to implement the training in a GP node (as described above):

# create a new array key for the model's prediction output (for when we test it)
prediction = gp.ArrayKey('PREDICTION')

# create a GP train node using our model, loss, and optimizer
train = gp.torch.Train(
    model,
    loss,
    optimizer,
    inputs = {
        'input': raw
    },
    loss_inputs = { # *** Q: What are loss inputs?? -> A: in GP's API notes
        0: prediction,
        1: seg
    },
    outputs = {
        0: prediction
    })
    
# Now add "train" to the pipeline along with everything else
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

# Finally we can build the pipeline and display it like normal:
with gp.build(pipeline):
    batch = pipeline.request_batch(request)
# include the prediction when calling imshow()
imshow(batch[raw].data, batch[seg].data, batch[prediction].data)

# To get better results, we can train the same model for a few more iterations:
with gp.build(pipeline):
    for i in range(10):
        batch = pipeline.request_batch(request)
        
imshow(batch[raw].data, batch[seg].data, batch[prediction].data)

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
f_in = zarr.open(zarr_dir)
f_out = zarr.open(output_zarr_dir, 'w')
f_out['raw'] = f_in['raw']
f_out['seg'] = f_in['seg']
ds = f_out.create_dataset('prediction', shape=(1,1,600,600))
ds.attrs['resolution'] = (1,1)
ds.attrs['offset'] = (0,0)

# create a ZarrWrite() node in order to store the predictions
zarr_write = gp.ZarrWrite(
    output_filename = output_zarr_dir,
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
    zarr.open(output_zarr_dir)['raw'][:],
    None,
    zarr.open(output_zarr_dir)['prediction'][:])

