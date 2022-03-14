import os
import argparse
import zarr
import torch

import utils.mgmodel as mgmodel

class LoadSetupAction(argparse.Action):
    """Load training setup from command line arg."""

    def __call__(self, parser, namespace, setup_dir, option_string=None):
        try:
            if os.path.exists(setup_dir) and \
                    os.path.isdir(setup_dir) and \
                    os.path.exists(os.path.join(setup_dir, 'train.py')):

                module = __import__(f'{setup_dir}.train')
                setattr(namespace, 'setup_dir', setup_dir)
                setattr(namespace, 'train', module.train.train)
                setattr(namespace, 'test', module.train.test)
        except:
            raise
            parser.error(f'Invalid setup: {setup_dir}')

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
                model = mgmodel.initModel()
            setattr(namespace, 'model', model)
            setattr(namespace, 'model_path', model_path)
        except FileNotFoundError:
            parser.error(f'File not found: {model_path}')
        except:
            raise
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
    parser.add_argument('setup', metavar='setup',
        action=LoadSetupAction,
        help='Path to directory containing requested training setup.')
    parser.add_argument('zarr', metavar='zarr',
        action=LoadZarrAction,
        help='Zarr directory containing input data and segmentation layers.')
    parser.add_argument('model', metavar='model.pth', default='model.pth',
        action=LoadModelAction,
        help="Specify a .pth file to use as the initial model. If it doesn't exist, it will be initialized.")
    parser.add_argument('-w', '--weights', metavar='pth',
        action=LoadModelWeightsAction,
        help='Optional: specify a .pth file to use as the initial model weights.')
    parser.add_argument('-t', metavar='type', choices=['test','train','both'],
        default='train', dest='type',
        help='Optional: specify whether to train or test the model. ' +
            'Possible values: test, train, or both. Default is train. ' +
            'Unless both, only the specified routine will be executed.')

    return parser.parse_args()
