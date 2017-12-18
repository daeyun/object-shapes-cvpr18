import gc
import threading
import queue
import time
from mvshape import benchmark
from os import path
from mvshape import io_utils
import sys
import numpy as np
from mvshape.models import encoder
from mvshape.models import decoder
from mvshape.models import encoderdecoder
from mvshape import torch_utils
import torchvision
import torch
import textwrap
from torch import optim
from dshin import log
from torch import nn
import argparse
from torch.autograd import Variable
from mvshape.data import dataset

torch.backends.cudnn.benchmark = True


def shrec12_convert_category_ids(ids):
    new_ids = dataset.shrec12_train_category_ids.searchsorted(ids)
    return new_ids


def to_numpy(gpu_var):
    return gpu_var.data.cpu().numpy()


parser = argparse.ArgumentParser(description='training')
parser.add_argument('--experiment', type=str)
parser.add_argument('--save_name', type=str)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--model_name', type=str)  # e.g. shapenetcore_rgb_mv6
parser.add_argument('--print_every', type=int, default=200)
# parser.add_argument('--debug', action='store_true', default=False)
args = parser.parse_args()


# is_rgb = True
# is_shrec12 = False
# experiment_name = 'vpo'
# experiment_name = 'opo'


def build_model(model_name):
    assert model_name in encoderdecoder.model_names, model_name
    # Build model with initial parameters.
    if model_name == 'shapenetcore_rgb_mv6':
        batch_size = 130
        models = encoderdecoder.build_multiview_model(name=model_name, train_batch_size=batch_size)
    elif model_name == 'shapenetcore_rgb_voxels':
        batch_size = 130
        models = encoderdecoder.build_multiview_model(name=model_name, train_batch_size=batch_size)
    elif model_name == 'shrec12_depth_mv6':
        batch_size = 150
        models = encoderdecoder.build_multiview_model(name=model_name, train_batch_size=batch_size)
    elif model_name == 'shrec12_voxels':
        batch_size = 150
        models = encoderdecoder.build_multiview_model(name=model_name, train_batch_size=batch_size)
    else:
        batch_size = 150
        raise NotImplementedError()
    torch_utils.recursive_module_apply(models, lambda m: m.cuda())
    return models


def load_model(resume_filename, model_name):
    assert model_name in encoderdecoder.model_names, model_name
    models = encoderdecoder.load_model(resume_filename, use_cpu=False)
    assert models['metadata']['name'] == model_name
    torch_utils.recursive_module_apply(models, lambda m: m.cuda())
    return models


def select_data_loader(model_name, experiment_name, batch_size):
    if model_name == 'shapenetcore_rgb_mv6':
        if experiment_name == 'vpo':
            # loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_{}/subset_examples.cbor'.format(experiment_name)),
            #                                 tensors_to_read=('input_rgb', 'target_depth'), shuffle=True, batch_size=batch_size)
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_rgb', 'target_depth'), shuffle=True, batch_size=batch_size)
        elif experiment_name == 'opo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_rgb', 'target_depth'), shuffle=True, batch_size=batch_size)
        else:
            raise NotImplementedError()
    elif model_name == 'shapenetcore_rgb_voxels':
        if experiment_name == 'vpo':
            # TODO#########################
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_{}/subset_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_rgb', 'target_voxels'), shuffle=True, batch_size=batch_size)
        elif experiment_name == 'opo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_rgb', 'target_voxels'), shuffle=True, batch_size=batch_size)
        else:
            raise NotImplementedError()
    elif model_name == 'shrec12_depth_mv6':
        if experiment_name == 'vpo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shrec12_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_depth', 'target_depth'), shuffle=True, batch_size=batch_size)
        elif experiment_name == 'opo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shrec12_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_depth', 'target_depth'), shuffle=True, batch_size=batch_size)
        else:
            raise NotImplementedError()
    elif model_name == 'shrec12_voxels':
        if experiment_name == 'vpo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shrec12_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_depth', 'target_voxels'), shuffle=True, batch_size=batch_size)
        elif experiment_name == 'opo':
            loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shrec12_examples_{}/all_examples.cbor'.format(experiment_name)),
                                            tensors_to_read=('input_depth', 'target_voxels'), shuffle=True, batch_size=batch_size)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    total_num_examples = loader.total_num_examples
    log.info('Dataset size: {}'.format(total_num_examples))
    return loader


def get_learning_rate(model_name):
    if model_name == 'shapenetcore_rgb_mv6':
        learning_rate = 0.0001
    elif model_name == 'shapenetcore_rgb_voxels':
        learning_rate = 0.0001
    elif 'shrec12' in model_name:
        learning_rate = 0.0001
    else:
        raise NotImplementedError()
    return learning_rate


def main():
    experiment_name = args.experiment.strip()
    save_name = args.save_name
    model_name = args.model_name
    resume_filename = path.expanduser(args.resume)
    print_every = args.print_every

    # sanity checks
    assert experiment_name in ('opo', 'vpo'), experiment_name
    assert save_name, save_name
    assert model_name in encoderdecoder.model_names, model_name
    assert print_every > 0

    learning_rate = get_learning_rate(model_name)

    log.info(args)

    if resume_filename:
        assert resume_filename.endswith('.pth')
        assert path.isfile(resume_filename)
        assert '/' + experiment_name in resume_filename  # sanity check. doesnt necessarily have to be true
        # e.g. '/data/mvshape/out/pytorch/shapenetcore_rgb_mv6/vpo/0/models0_00000_0017500_00017500.pth'
        models = load_model(resume_filename, model_name)
    else:
        models = build_model(model_name)

    # models['metadata']['batch_size']
    batch_size = models['metadata']['batch_size']

    # output of model is
    # ((b, 1, 128, 128), (b, 1, 128, 128))

    # TODO: this should probably be in gpu
    helper_torch_modules = encoderdecoder.build_helper_torch_modules()

    all_params = []
    torch_utils.recursive_module_apply(models, lambda m: all_params.extend(m.parameters()))
    assert len(all_params) > 0
    optimizer = optim.Adam(all_params, lr=learning_rate)
    log.info('Model is ready. Loading training metadata.')

    # Initialize dataset.
    #####################################
    loader = select_data_loader(model_name=model_name, experiment_name=experiment_name, batch_size=batch_size)
    log.info('Data is ready.')

    save_dir = path.join(dataset.base, 'out/pytorch/{}/{}/{}/'.format(model_name, experiment_name, save_name))

    # Main loop
    while True:
        encoderdecoder.train_epoch(
            models=models,
            data_loader=loader,
            optimizer=optimizer,
            helper_torch_modules=helper_torch_modules,
            save_every=2500,
            print_every=print_every,
            save_dir=save_dir,
        )


if __name__ == '__main__':
    main()
