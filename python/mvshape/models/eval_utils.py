"""
Code related to integrating datasets and specific models in `models` subpackage.
"""
import typing
import matplotlib.pyplot as pt
import itertools
import torch
import torchvision
import collections
import numpy as np
from mvshape import torch_utils
from mvshape.proto import dataset_pb2
from mvshape.data import dataset
from mvshape import shapes

from dshin import geom2d


def load_torch_model(filename, use_cpu=True) -> dict:
    """
    Returns a dict of pytorch modules. They are (somewhat poorly) named "models."
    :param filename: e.g. /data/mvshape/out/pytorch/mv6_vpo/models5_0050.pth
    :param use_cpu: If True, the model is loaded in cpu mode.
    :return: A dict of pytorch models that make up a neural net.
    """
    with open(filename, 'rb') as f:
        models = torch.load(f, map_location=lambda storage, loc: storage)

    if use_cpu:
        def set_device(module):
            return module.cpu()
    else:
        def set_device(module):
            return module.cuda()
    models = torch_utils.recursive_module_apply(models, func=set_device)

    return models


def get_model_batch_output(models: dict, next_batch: dict, use_cpu=True) -> dict:
    """
    :param models: Dict of pytorch models that make up one network.
                   Must be in `.eval()` mode.
    :param next_batch: Output from `ExampleLoader.next()`. May be None.
    :return: Dict of all output as numpy arrays or Example protobufs.
    """
    assert next_batch is not None
    assert len(next_batch) == 2
    assert isinstance(next_batch[0], (list, tuple))
    assert isinstance(next_batch[1], dict)
    batch_data = dataset.prepare_data(next_batch)

    # Only used as a return value.
    examples = next_batch[0]
    assert isinstance(examples, (list, tuple)) and len(examples) > 0
    assert isinstance(examples[0], dataset_pb2.Example)

    # (b, 2, 128, 128)
    in_image_np = batch_data['in_image']
    in_image = torch_utils.recursive_numpy_to_torch(in_image_np, cuda=not use_cpu)

    h = models['encoder'](in_image)
    out_images = []
    target_images = []

    # repeat the each "branch" and collect output.
    for i in range(len(models['h_shared'])):
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

        out_silhouette = torch_utils.recursive_torch_to_numpy(out_silhouette)
        out_depth_front = torch_utils.recursive_torch_to_numpy(out_depth_front)
        out_depth_back = torch_utils.recursive_torch_to_numpy(out_depth_back)

        # Flip the "back" images horizontally.
        out_front_masked = dataset.to_masked_image(out_depth_front, out_silhouette)
        out_back_masked = dataset.to_masked_image(out_depth_back, out_silhouette)[:, :, :, ::-1]
        target_front_masked = dataset.to_masked_image(target_depth_front, target_silhouette)
        target_back_masked = dataset.to_masked_image(target_depth_back, target_silhouette)[:, :, :, ::-1]

        # This should be the same as the order specified in the dataset.
        out_images.append(out_front_masked)
        out_images.append(out_back_masked)
        target_images.append(target_front_masked)
        target_images.append(target_back_masked)

    # important to subtract the offset value.
    out_images = np.concatenate(out_images, axis=1) - dataset.shrec12_target_depth_offset
    target_images = np.concatenate(target_images, axis=1)

    in_image_np_masked = dataset.to_masked_image(in_image_np[:, 0, None], in_image_np[:, 1, None])

    # masked with nan values for convenience.
    return {
        'out_images': out_images,
        'in_images': in_image_np_masked,
        'target_images': target_images,
        'examples': examples,
    }


def _load_all_eval_examples_shrec12(pytorch_saved_model_filename: str, metadata_filename: str, use_cpu=True) -> typing.Dict[str, typing.Dict[str, np.ndarray]]:
    """
    :param pytorch_saved_model_filename: e.g. /data/mvshape/out/pytorch/mv6_vpo/models5_0050.pth
    :param metadata_filename:  e.g. '/data/mvshape/out/splits/shrec12_examples_vpo/test.bin'
    :return:
    Example usage of output:
        out = ...
        assert out['NOVELVIEW']['out_images'].shape == (b, 1, 128, 128)
        assert np.isnan(out['NOVELVIEW']['out_images']).any()

    """

    models = load_torch_model(pytorch_saved_model_filename, use_cpu=use_cpu)
    # Convert to eval mode, if not already in eval mode.
    torch_utils.recursive_module_apply(models, lambda m: m.eval())

    # A constant for now. The entire dataset is returned, so this can be any number that fits in memory.
    batch_size = 50
    subset_tags = ['NOVELVIEW', 'NOVELMODEL', 'NOVELCLASS']

    all_out_dict = {}
    for subset_tag in subset_tags:
        subset_output_list = []
        eval_loader = dataset.ExampleLoader(metadata_filename,
                                            tensors_to_read=['single_depth', 'multiview_depth', ],
                                            batch_size=batch_size,
                                            subset_tags=[subset_tag],
                                            shuffle=False)

        while True:
            # Returns None at the end of epoch.
            next_batch = eval_loader.next()
            if next_batch is None:
                break
            batch_output = get_model_batch_output(models, next_batch, use_cpu=use_cpu)  # type: dict
            subset_output_list.append(batch_output)

        assert len(subset_output_list) > 0

        concatenated_out_dict = {}
        keys = list(subset_output_list[0].keys())
        for key in keys:
            out_items = []
            for batch_output in subset_output_list:
                out = batch_output[key]
                out_items.append(out)

            # Reduce
            if isinstance(out_items[0], np.ndarray):
                concatenated_out_dict[key] = np.concatenate(out_items)
            elif isinstance(out_items[0], (list, tuple)):
                assert isinstance(out_items[0][0], dataset_pb2.Example)
                concatenated_out_dict[key] = list(itertools.chain(*out_items))
            else:
                raise NotImplementedError('Unknown return value type: {}'.format(type(out_items[0])))

        all_out_dict[subset_tag] = concatenated_out_dict

    return all_out_dict


def load_all_eval_examples_shrec12(pytorch_saved_model_filename: str, metadata_filename: str, use_cpu=True):
    all_out_dict = _load_all_eval_examples_shrec12(pytorch_saved_model_filename=pytorch_saved_model_filename,
                                                   metadata_filename=metadata_filename,
                                                   use_cpu=use_cpu)
    return shapes.MVShapeResultsShrec12(all_out_dict)
