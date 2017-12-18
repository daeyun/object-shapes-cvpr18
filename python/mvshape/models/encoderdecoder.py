import gc
import pprint
import threading
import queue
import time
from mvshape import benchmark
from os import path
from mvshape import io_utils
import sys
from mvshape.data import dataset
import numpy as np
from mvshape.models import encoder
from mvshape.models import decoder
from mvshape.models import eval_utils
from mvshape import torch_utils
import torchvision
import torch
import textwrap
from torch import optim
from torch import nn
import argparse
from dshin import log
from torch.autograd import Variable
import scipy.special
from mvshape.data import dataset

experiment_names = ['vpo', 'opo']
# model_names = ['shapenetcore_rgb_mv6', 'shrec12_depth']
model_names = ['shapenetcore_rgb_mv6', 'shrec12_voxels', 'shapenetcore_rgb_voxels', 'shrec12_depth_mv6']


def build_multiview_model(name, train_batch_size):
    assert train_batch_size > 1
    if name == 'shapenetcore_rgb_mv6':
        num_input_channels = 3
        num_views = 6
        layers = [4, 7, 4]

        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, layers, input_channel=num_input_channels),
            'decoder_depth': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'decoder_silhouette': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'h_shared': [],
            'h_depth_front': [],
            'h_depth_back': [],
            'h_silhouette': [],
            'silhouette_loss_weight': 0.2,
            'metadata': {
                'name': name,
                'global_step': 0,
                'current_epoch': -1,  # incremented in the beginning. so -1 becomes 0.
                'current_epoch_step': 0,
                'batch_size': train_batch_size,
            }
        }

        for i in range(num_views // 2):
            models['h_shared'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_depth_front'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_depth_back'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_silhouette'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))

    elif name == 'shrec12_depth_mv6':
        num_input_channels = 2
        num_views = 6
        layers = [3, 4, 3]

        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, layers, input_channel=num_input_channels),
            'decoder_depth': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'decoder_silhouette': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'h_shared': [],
            'h_depth_front': [],
            'h_depth_back': [],
            'h_silhouette': [],
            'silhouette_loss_weight': 0.2,
            'metadata': {
                'name': name,
                'global_step': 0,
                'current_epoch': -1,  # incremented in the beginning. so -1 becomes 0.
                'current_epoch_step': 0,
                'batch_size': train_batch_size,
            }
        }

        for i in range(num_views // 2):
            models['h_shared'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_depth_front'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_depth_back'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
            models['h_silhouette'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))


    elif name == 'shrec12_voxels':
        num_input_channels = 2
        num_views = None
        layers = [3, 4, 3]

        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, layers, input_channel=num_input_channels),
            'decoder': decoder.Decoder3d_48(in_size=2048, out_channels=1),
            'h_shared': [],
            'metadata': {
                'name': name,
                'global_step': 0,
                'current_epoch': -1,  # incremented in the beginning. so -1 becomes 0.
                'current_epoch_step': 0,
                'batch_size': train_batch_size,
            }
        }

        # "shared" by 1 "view" that is the (48,48,48) voxels. See this as part of the decoder.
        models['h_shared'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))

    elif name == 'shapenetcore_rgb_voxels':
        num_input_channels = 3
        layers = [4, 7, 4]

        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, layers, input_channel=num_input_channels),
            'decoder': decoder.Decoder3d_32(in_size=2048, out_channels=1),
            'metadata': {
                'name': name,
                'global_step': 0,
                'current_epoch': -1,  # incremented in the beginning. so -1 becomes 0.
                'current_epoch_step': 0,
                'batch_size': train_batch_size,
            }
        }

    else:
        raise NotImplementedError()

    return models


def determine_num_views_from_model_dict(models: dict) -> int:
    """
    returns `num_views` (6 or 20 for mv) from model dict.
    """
    if 'h_shared' in models and 'h_depth_front' in models:
        # front/back depths sharing one silhouette.
        num_views = len(models['h_shared']) * 2
    else:
        raise NotImplementedError()
    return num_views


def get_target_depth_offset(model_name):
    if 'shapenetcore' in model_name:
        target_depth_offset = dataset.shapenetcore_target_depth_offset
    elif 'shrec12' in model_name:
        target_depth_offset = dataset.shrec12_target_depth_offset
    else:
        raise NotImplementedError()
    return target_depth_offset


def build_helper_torch_modules():
    """ torch modules for loss functions etc. to compute in gpu
    """
    helper_torch_modules = {
        'classification': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),  # for depth
        'bce': nn.BCELoss(),  # for silhouette
        'sigmoid': nn.Sigmoid(),
    }
    return helper_torch_modules


def is_model_cuda(models: dict):
    # assuming there's always a module named "encoder"
    # also assuming if encoder is in gpu, the other modules are too.
    return next(models['encoder'].parameters()).is_cuda


def get_output_from_model(models: dict, batch_data: dict, helper_torch_modules: dict, include_output_tensors: bool):
    """
    :param include_output_tensors: Whether output images or voxels are included in the output dict. Typically not used during training.
    :param helper_torch_modules: Output of build_helper_torch_modules. Reused.
    :param models:
    :param batch_data: output of prepare_*
    :return: A dict of torch Variables.
    """

    # We assume at least `in_image` is present in all cases.
    model_input_image = batch_data['in_image']
    assert isinstance(model_input_image, Variable)

    ret = {}  # return values are stored here.

    # shapenetcore_rgb_mv6, etc.
    if '_mv' in models['metadata']['name']:
        # multiview. front/back model only.
        # assert model_input_image
        h = models['encoder'](model_input_image)  # shared by all views

        num_views = determine_num_views_from_model_dict(models)
        assert num_views >= 2

        silhouette_losses = []
        depth_losses = []
        front_depth_losses = []
        back_depth_losses = []
        total_losses = []
        silhouette_ious = []

        output_tensors = {
            'depth_front': [],
            'depth_back': [],
            'silhouette': [],
        }

        for i in range(num_views // 2):
            # shared by front and back
            h_i = models['h_shared'][i](h)
            h_i_df = models['h_depth_front'][i](h_i)
            h_i_db = models['h_depth_back'][i](h_i)
            h_i_s = models['h_silhouette'][i](h_i)

            # output images
            out_depth_front = models['decoder_depth'](h_i_df)  # (b, 1, 128, 128)
            out_depth_back = models['decoder_depth'](h_i_db)  # (b, 1, 128, 128)
            out_silhouette = models['decoder_silhouette'](h_i_s)  # (b, 1, 128, 128)

            # target images
            target_depth_front = batch_data['target_depth'][:, i * 2]  # (b, 1, 128, 128)
            target_depth_back = batch_data['target_depth'][:, i * 2 + 1]  # (b, 1, 128, 128) already flipped
            target_silhouette = batch_data['target_silhouette'][:, i]  # (b, 1, 128, 128)

            # compute losses
            target_depth_offset = get_target_depth_offset(model_name=models['metadata']['name'])
            depth_loss_i_front = compute_masked_depth_loss(out_depth_front, target_depth_front, target_silhouette, target_depth_offset=target_depth_offset)
            depth_loss_i_back = compute_masked_depth_loss(out_depth_back, target_depth_back, target_silhouette, target_depth_offset=target_depth_offset)
            depth_loss_i = (depth_loss_i_front + depth_loss_i_back) * 0.5

            silhouette_loss_i = compute_silhouette_loss(out_silhouette, target_silhouette, helper_torch_modules=helper_torch_modules)
            total_loss_i = depth_loss_i + models['silhouette_loss_weight'] * silhouette_loss_i
            # total_loss_i is a torch Variable and its .data is FloatTensor, either on gpu or cpu.

            # silhouette iou
            silhouette_iou_i = compute_silhouette_iou(out_silhouette, target_silhouette=target_silhouette)

            front_depth_losses.append(depth_loss_i_front)
            back_depth_losses.append(depth_loss_i_back)

            total_losses.append(total_loss_i)
            silhouette_losses.append(silhouette_loss_i)
            depth_losses.append(depth_loss_i)

            silhouette_ious.append(silhouette_iou_i)

            if include_output_tensors:
                output_tensors['depth_front'].append(out_depth_front)
                output_tensors['depth_back'].append(out_depth_back)
                output_tensors['silhouette'].append(out_silhouette)

        # mean over a single batch.
        ret['loss'] = {
            'silhouette': torch_utils.reduce_mean(silhouette_losses),
            'depth_front': torch_utils.reduce_mean(front_depth_losses),
            'depth_back': torch_utils.reduce_mean(back_depth_losses),
            'depth': torch_utils.reduce_mean(depth_losses),  # mean of front and back losses.
            'total': torch_utils.reduce_mean(total_losses),
        }

        # ret['loss']['total'] should always be the final loss. trainer minimizes it.

        # does not have to be differentiable.
        ret['metric'] = {
            'silhouette_iou': torch_utils.reduce_mean(silhouette_ious)
        }

        if include_output_tensors:
            ret['output_tensor'] = output_tensors
    elif '_voxels' in models['metadata']['name']:
        h = models['encoder'](model_input_image)

        # see this as part of the decoder

        h2 = models['h_shared'][0](h)
        out_voxels = models['decoder'](h2)
        voxel_loss = compute_voxel_loss(out_voxels, batch_data['target_voxels'], helper_torch_modules)
        voxel_iou = compute_voxel_iou(out_voxels, batch_data['target_voxels'])

        ret['loss'] = {
            'total': voxel_loss,
        }
        ret['metric'] = {
            'voxel_iou': voxel_iou,
        }

        if include_output_tensors:
            ret['output_tensor'] = {
                'voxels': out_voxels
            }

    else:
        raise NotImplementedError()

    # asserts `ret` has variables only. no side effect.
    torch_utils.recursive_variable_apply(ret, lambda v: None)

    return ret


def load_model(filename, use_cpu=True):
    assert filename.endswith('.pth'), filename
    parts = path.basename(filename.split('.')[0]).split('_')
    current_epoch = int(parts[1])
    current_epoch_step = int(parts[2])
    global_step = int(parts[3])
    model = eval_utils.load_torch_model(filename, use_cpu=use_cpu)
    log.info('Loading model {}'.format((current_epoch, current_epoch_step, global_step)))
    assert 'metadata' in model

    # if 'metadata' in model:
    #     assert model['metadata']['global_step'] == global_step
    #     assert model['metadata']['current_epoch'] == current_epoch
    #     assert model['metadata']['current_epoch_step'] == current_epoch_step
    # else:
    #     model['metadata'] = {
    #         'global_step': global_step,
    #         'current_epoch': current_epoch,
    #         'current_epoch_step': current_epoch_step,
    #     }
    return model


def sigmoid(x):
    # overflows
    # return 1.0 / (1.0 + np.exp(-x))

    return scipy.special.expit(x)


def interweave_images(a, b, axis=0):
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    if axis == 0:
        newsize = (a.shape[0] * 2,) + a.shape[1:]
        ret = np.empty(newsize, dtype=a.dtype)
        ret[0::2] = a
        ret[1::2] = b
    elif axis == 1:
        newsize = (a.shape[0], a.shape[1] * 2,) + a.shape[2:]
        ret = np.empty(newsize, dtype=a.dtype)
        ret[:, 0::2] = a
        ret[:, 1::2] = b
    else:
        raise NotImplementedError()

    return ret


def apply_masks_to_front_back_depths(target_depth, target_silhouette):
    """
    e.g. in the order of front_0, back_0_flipped, front_1, back_1_flipped, front_2, back_2_flipped
    returns unflipped and masked images
    """
    assert target_depth.shape[0] == target_silhouette.shape[0] * 2
    assert target_silhouette.dtype == np.bool
    d = target_depth.copy().reshape(-1, 128, 128)
    s = target_silhouette.copy().reshape(-1, 128, 128)
    d[::2][~s] = np.nan
    d[1::2][~s] = np.nan

    d[1::2] = d[1::2, :, ::-1]

    return d


def get_final_images_from_model(models, im, helper_torch_modules):
    """
    Returns an array of shape (v=6, 128, 128).
    """
    if 'shrec12' in models['metadata']['name']:
        assert im.shape[1:] == (2, 128, 128)
    else:
        assert im.shape[1:] == (3, 128, 128)
        # max-min <= 1.0
        assert np.ptp(im) <= 1.0
    assert im.dtype == np.float32

    target_depth_offset = get_target_depth_offset(model_name=models['metadata']['name'])
    num_views = determine_num_views_from_model_dict(models)
    assert num_views % 2 == 0
    is_cuda = is_model_cuda(models)

    im_batch_size = im.shape[0]

    batch_data_np = {}
    batch_data_np['in_image'] = im

    # dummy values.
    batch_data_np['target_depth'] = np.ones((im_batch_size, num_views, 1, 128, 128), dtype=np.float32)
    batch_data_np['target_silhouette'] = np.ones((im_batch_size, num_views // 2, 1, 128, 128), dtype=np.float32)

    batch_data = torch_utils.recursive_numpy_to_torch(batch_data_np, cuda=is_cuda)
    out = get_output_from_model(models, batch_data, helper_torch_modules=helper_torch_modules, include_output_tensors=True)

    out = torch_utils.recursive_torch_to_numpy(out)

    # assuming "depth_front" exists. it's easy to change
    assert len(out['output_tensor']['silhouette']) == len(out['output_tensor']['depth_front'])

    # size of out['output_tensor']['silhouette'] is num_views/2. Each item looks like (b, 1, 128, 128)

    silhouette_prob = np.concatenate([sigmoid(out['output_tensor']['silhouette'][i].reshape(im_batch_size, 1, 128, 128)) for i in range(num_views // 2)], axis=1)
    depth_fronts = np.concatenate([out['output_tensor']['depth_front'][i].reshape(im_batch_size, 1, 128, 128) for i in range(num_views // 2)], axis=1)
    depth_backs = np.concatenate([out['output_tensor']['depth_back'][i].reshape(im_batch_size, 1, 128, 128) for i in range(num_views // 2)], axis=1)

    threshold = 0.4

    df = depth_fronts.copy()
    db = depth_backs.copy()

    df[silhouette_prob < threshold] = np.nan
    db[silhouette_prob < threshold] = np.nan

    # flip
    db = db[:, :, :, ::-1]

    masked_depth = interweave_images(df, db, axis=1)

    masked_depth -= target_depth_offset

    return {
        'masked_depth': masked_depth,
        'silhouette_prob': silhouette_prob,
    }


def prepare_data_rgb_mv(next_batch):
    """
    :param next_batch: Output from data loader.
    :return: A dict of (field -> numpy array)
    """

    # A list of Example protobufs and a dict of tensors.
    examples, tensors = next_batch

    # prepare data
    # modify in-place.
    # if is_rgb:
    #     in_rgb = tensors['input_rgb'].astype(np.float32) / 255.0
    #     in_image = in_rgb
    # else:
    #     in_depth = tensors['input_depth'].copy()
    #     in_silhouette = np.isfinite(in_depth)
    #     in_depth[~in_silhouette] = 0
    #     in_silhouette = in_silhouette.astype(np.float32)
    #     in_image = np.concatenate([in_depth, in_silhouette], axis=1)
    #     assert (in_image.ndim == 4 and in_image.shape[1] == 2), in_image.shape
    in_rgb = tensors['input_rgb'].astype(np.float32) / 255.0
    in_image = in_rgb

    target_depth = tensors['target_depth'].copy()
    target_silhouette = np.isfinite(target_depth)
    target_depth[~target_silhouette] = 0
    target_silhouette = target_silhouette.astype(np.float32)  # 0 or 1
    # both (b, 6, 128, 128)

    target_depth = np.expand_dims(target_depth, axis=2)
    target_silhouette = np.expand_dims(target_silhouette, axis=2)
    # both (b, 6, 1, 128, 128)

    ## config for the front-back depth model. flip the back depth images.
    target_depth[:, 1::2] = target_depth[:, 1::2, :, :, ::-1]
    # (b, 6, 1, 128, 128)
    target_silhouette = target_silhouette[:, ::2]
    # (b, 3, 1, 128, 128)

    # categories = np.array(shrec12_convert_category_ids([example.single_depth.category_id - 1 for example in examples]), dtype=np.int64)

    ret = {
        'in_image': in_image,
        'target_depth': target_depth,
        'target_silhouette': target_silhouette,
    }

    # 'categories': categories,

    return ret


def prepare_data_depth_mv(next_batch):
    """
    :param next_batch: Output from data loader.
    :return: A dict of (field -> numpy array)
    """

    # A list of Example protobufs and a dict of tensors.
    examples, tensors = next_batch

    # prepare data
    # modify in-place.
    in_depth = tensors['input_depth'].copy()
    in_silhouette = np.isfinite(in_depth)
    in_depth[~in_silhouette] = 0
    in_silhouette = in_silhouette.astype(np.float32)
    in_image = np.concatenate([in_depth, in_silhouette], axis=1)
    assert (in_image.ndim == 4 and in_image.shape[1] == 2), in_image.shape

    target_depth = tensors['target_depth'].copy()
    target_silhouette = np.isfinite(target_depth)
    target_depth[~target_silhouette] = 0
    target_silhouette = target_silhouette.astype(np.float32)  # 0 or 1
    # both (b, 6, 128, 128)

    target_depth = np.expand_dims(target_depth, axis=2)
    target_silhouette = np.expand_dims(target_silhouette, axis=2)
    # both (b, 6, 1, 128, 128)

    ## config for the front-back depth model. flip the back depth images.
    target_depth[:, 1::2] = target_depth[:, 1::2, :, :, ::-1]
    # (b, 6, 1, 128, 128)
    target_silhouette = target_silhouette[:, ::2]
    # (b, 3, 1, 128, 128)

    # categories = np.array(shrec12_convert_category_ids([example.single_depth.category_id - 1 for example in examples]), dtype=np.int64)

    ret = {
        'in_image': in_image,
        'target_depth': target_depth,
        'target_silhouette': target_silhouette,
    }

    # 'categories': categories,


    return ret


def prepare_data_depth_voxels(next_batch):
    """
    :param next_batch: Output from data loader.
    :return: A dict of (field -> numpy array)
    """

    # A list of Example protobufs and a dict of tensors.
    examples, tensors = next_batch

    # prepare data
    # modify in-place.
    in_depth = tensors['input_depth'].copy()
    in_silhouette = np.isfinite(in_depth)
    in_depth[~in_silhouette] = 0
    in_silhouette = in_silhouette.astype(np.float32)
    in_image = np.concatenate([in_depth, in_silhouette], axis=1)
    assert (in_image.ndim == 4 and in_image.shape[1] == 2), in_image.shape

    # categories = np.array(shrec12_convert_category_ids([example.single_depth.category_id - 1 for example in examples]), dtype=np.int64)
    target_voxels = tensors['target_voxels'].copy().astype(np.float32)

    ret = {
        'in_image': in_image,
        'target_voxels': target_voxels,
    }

    return ret


def prepare_data_rgb_voxels(next_batch):
    """
    :param next_batch: Output from data loader.
    :return: A dict of (field -> numpy array)
    """

    # A list of Example protobufs and a dict of tensors.
    examples, tensors = next_batch

    # prepare data
    # modify in-place.
    # if is_rgb:
    #     in_rgb = tensors['input_rgb'].astype(np.float32) / 255.0
    #     in_image = in_rgb
    # else:
    #     in_depth = tensors['input_depth'].copy()
    #     in_silhouette = np.isfinite(in_depth)
    #     in_depth[~in_silhouette] = 0
    #     in_silhouette = in_silhouette.astype(np.float32)
    #     in_image = np.concatenate([in_depth, in_silhouette], axis=1)
    #     assert (in_image.ndim == 4 and in_image.shape[1] == 2), in_image.shape
    in_rgb = tensors['input_rgb'].astype(np.float32) / 255.0
    in_image = in_rgb

    target_voxels = tensors['target_voxels'].copy().astype(np.float32)

    ret = {
        'in_image': in_image,
        'target_voxels': target_voxels,
    }

    # 'categories': categories,

    return ret


def save_model(model, save_dir):
    current_epoch = model['metadata']['current_epoch']
    current_epoch_step = model['metadata']['current_epoch_step']
    global_step = model['metadata']['global_step']

    filename = path.join(save_dir, 'models0_{:05}_{:07}_{:08}.pth'.format(current_epoch, current_epoch_step, global_step))

    io_utils.ensure_dir_exists(path.dirname(filename))

    with open(filename, 'wb') as f:
        log.info('Saving.. {}'.format(filename))
        torch.save(model, f)
        log.info('Saved.')


def compute_classification_accuracy(out, classes):
    predicted = out.max(dim=1)[1]
    num_correct = (predicted == classes).int().sum().cpu().data.numpy()[0]
    num_total = predicted.size(0)
    accuracy = float(num_correct) / num_total
    return accuracy, num_correct, num_total


def compute_masked_depth_loss(out_depth: Variable, target_depth: Variable, target_silhouette: Variable, target_depth_offset):
    target_depth_with_offset = (target_depth + target_depth_offset) * target_silhouette
    area = target_silhouette.float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) + 1e-3

    assert len(out_depth.size()) == 4
    assert len(target_depth.size()) == 4
    assert len(target_silhouette.size()) == 4

    masked_sq_sum = (torch.pow(out_depth - target_depth_with_offset, 2.0) * target_silhouette).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    out = (masked_sq_sum / area).mean()
    return out


def compute_silhouette_loss(out_silhouette: Variable, target_silhouette: Variable, helper_torch_modules):
    # area outside silhouette should be zero'ed out.
    out = helper_torch_modules['bce'](helper_torch_modules['sigmoid'](out_silhouette), target_silhouette)
    return out


def compute_voxel_loss(out_voxels: Variable, target_voxels: Variable, helper_torch_modules):
    # area outside silhouette should be zero'ed out.
    out = helper_torch_modules['bce'](helper_torch_modules['sigmoid'](out_voxels), target_voxels)
    return out


def compute_silhouette_iou(out_silhouette: Variable, target_silhouette: Variable):
    assert len(out_silhouette.size()) == 4
    assert len(target_silhouette.size()) == 4

    a = (out_silhouette >= 0).float()
    b = (target_silhouette >= 0.5).float()

    intersection = ((a + b) >= 2).float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
    union = ((a + b) >= 1).float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    out = (intersection / (union + 1e-7)).mean()
    return out


def compute_voxel_iou(out_voxels: Variable, target_voxels: Variable):
    assert len(out_voxels.size()) == 5
    assert len(target_voxels.size()) == 5

    a = (out_voxels >= 0).float()  # before sigmoid
    b = (target_voxels >= 0.5).float()

    intersection = ((a + b) >= 2).float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)
    union = ((a + b) >= 1).float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True).sum(dim=4, keepdim=True)

    out = (intersection / (union + 1e-7)).mean()
    return out


def data_queue_worker(data_loader: dataset.ExampleLoader2, data_queue: queue.Queue, models: dict, discard_uneven_last_batch: bool):
    """
    :param data_loader: Data source object. Implements next() and reset().
    :param data_queue:
    :param models: Only needed to access the metadata. Read-only.
    :param discard_uneven_last_batch: Should be False for evaluation mode. Often true for training.
    :return:
    """
    batch_size = models['metadata']['batch_size']
    while True:
        next_batch = data_loader.next()
        if next_batch is None or (discard_uneven_last_batch and (len(next_batch[0]) != batch_size)):
            log.info('Resetting data_loader')
            data_loader.reset()
            data_queue.put(None, block=True)
            break

        if models['metadata']['name'] == 'shapenetcore_rgb_mv6':
            batch_data_np = prepare_data_rgb_mv(next_batch)
        elif models['metadata']['name'] == 'shapenetcore_rgb_voxels':
            batch_data_np = prepare_data_rgb_voxels(next_batch)
        elif models['metadata']['name'] == 'shrec12_depth_mv6':
            batch_data_np = prepare_data_depth_mv(next_batch)
        elif models['metadata']['name'] == 'shrec12_voxels':
            batch_data_np = prepare_data_depth_voxels(next_batch)
        else:
            log.error('ERROR: Not implemented')
            sys.exit(1)

        # move to gpu
        batch_data = torch_utils.recursive_numpy_to_torch(batch_data_np, cuda=True)
        data_queue.put(batch_data, block=True)

    log.info('End of data queue')


def print_status(models, num_all_examples, io_block_times: list, accum_scalars):
    print('')
    log.info('    {:08d} of {:08d}. io block: {:.5f}s \n{}'.format(
        models['metadata']['current_epoch_step'],
        int(num_all_examples / models['metadata']['batch_size']),
        np.mean(io_block_times),
        pprint.pformat(torch_utils.recursive_dict_of_lists_mean(accum_scalars)),
    ))


def train_epoch(models,
                data_loader: dataset.ExampleLoader2,
                optimizer: torch.optim.Optimizer,
                helper_torch_modules: dict,
                save_every=2500,
                print_every=50,
                save_dir=None
                ):
    log.info('--- Started training. \n{}'.format(pprint.pformat(models['metadata'])))

    torch_utils.recursive_train_setter(models, is_training=True)
    gc.collect()

    batch_size = models['metadata']['batch_size']
    num_all_examples = data_loader.total_num_examples

    assert batch_size == data_loader.batch_size, (batch_size, data_loader.batch_size)

    # sanity check. not really necessary. just making sure .reset() was called.
    assert data_loader.current_batch_index == 0, data_loader.current_batch_index

    current_num_examples = 0
    all_scalar_outputs = {}

    data_queue = queue.Queue(maxsize=1)

    thread = threading.Thread(target=data_queue_worker, args=[data_loader, data_queue, models, True], daemon=True)
    thread.start()

    io_block_times = []

    model_name = models['metadata']['name']

    models['metadata']['current_epoch'] += 1
    log.info('\n### New epoch: {}\n'.format(models['metadata']['current_epoch']))

    models['metadata']['current_epoch_step'] = 0

    while True:
        start_time = time.time()
        batch_data = data_queue.get(block=True)
        io_block_times.append(time.time() - start_time)

        if models['metadata']['current_epoch_step'] > 0:
            if models['metadata']['current_epoch_step'] % print_every == 0:
                print_status(models, num_all_examples, io_block_times, accum_scalars=all_scalar_outputs)

            if save_dir is not None and models['metadata']['current_epoch_step'] % save_every == 0:
                # save_model(models, save_dir=path.join(dataset.base, 'out/pytorch/{}/{}/{}/'.format(model_name, experiment_name, save_name)))
                save_model(models, save_dir=save_dir)

        # Check if end of epoch.
        if batch_data is None:
            print_status(models, num_all_examples, io_block_times, accum_scalars=all_scalar_outputs)
            if save_dir is not None:
                save_model(models, save_dir=save_dir)
            break

        assert batch_data['in_image'].size(0) == batch_size

        optimizer.zero_grad()

        model_output = get_output_from_model(models,
                                             batch_data=batch_data,
                                             helper_torch_modules=helper_torch_modules,
                                             include_output_tensors=False)
        total_loss = model_output['loss']['total']

        # modifies `all_scalar_outputs` in-place
        torch_utils.recursive_scalar_dict_merge({
            'loss': model_output['loss'],
            'metric': model_output['metric'],
        }, all_scalar_outputs)

        # calling `.backward()` seems to trigger garbage collection. without it, it runs out of memory during testing.
        # this doesn't update any parameters, so it's safe to run when `is_training` is false too.
        total_loss.backward()
        optimizer.step()

        # running `to_python_scalar` triggers a sync point.
        # total_loss_sum += (to_python_scalar(total_loss) * batch_data['in_image'].size(0)) / this_dataset_total_num_examples
        current_num_examples += batch_size
        models['metadata']['current_epoch_step'] += 1
        models['metadata']['global_step'] += 1
        print('.', end='', flush=True)

    log.info('\nTrained {} examples\n\n'.format(num_all_examples))
    return torch_utils.recursive_dict_of_lists_mean(all_scalar_outputs)
