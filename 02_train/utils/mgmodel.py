import torch
from funlib.learn.torch.models import UNet, ConvPass

def initModel():
    num_fmaps = 8
    # Define the UNet
    unet = UNet(
        in_channels=3,
        num_fmaps=num_fmaps, # tutorial used 4
        fmap_inc_factor=2,
        downsample_factors=[[2,2], [2,2]],
        kernel_size_down=[[[3,3], [3,3]]]*3,
        kernel_size_up=[[[3,3], [3,3]]]*2,
        padding='same')
        
    # Define the model
    model = torch.nn.Sequential(
        unet,
        ConvPass(in_channels=num_fmaps,
                 out_channels=1,
                 kernel_sizes=[(1,1)], # 2 dimensions => Conv2d
                 activation=None),
        torch.nn.Sigmoid())
    
    return model
