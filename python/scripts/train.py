import gc
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
import torchvision
import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
from mvshape.data import dataset

torch.backends.cudnn.benchmark = True


def shrec12_convert_category_ids(ids):
    new_ids = dataset.shrec12_train_category_ids.searchsorted(ids)
    return new_ids


def to_python_scalar(gpu_var):
    d = gpu_var.cpu().data.numpy()
    assert len(d) == 1
    return d[0]


def to_numpy(gpu_var):
    return gpu_var.data.cpu().numpy()


def recursive_train_setter(module, is_training: bool):
    if isinstance(module, (list, tuple)):
        for item in module:
            recursive_train_setter(item, is_training)
    elif isinstance(module, dict):
        for item in module.values():
            recursive_train_setter(item, is_training)
    elif isinstance(module, nn.Module):
        if is_training:
            module.train()
        else:
            module.eval()
    else:
        raise RuntimeError('unknown module {}'.format(module))


is_rgb = True
is_shrec12 = False


def main():
    if is_rgb:
        batch_size = 150
        learning_rate = 0.0002
    else:
        batch_size = 150
        learning_rate = 0.0005
    silhouette_loss_weight = 0.2
    num_views = 6

    # specific to shrec12 for now.
    if is_shrec12:
        target_depth_offset = dataset.shrec12_target_depth_offset
    else:
        target_depth_offset = dataset.shapenetcore_target_depth_offset

    # globals
    current_epoch = 0

    if is_rgb:
        # Initialize network.
        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, [4, 7, 4], input_channel=3 if is_rgb else 2),
            'decoder_depth': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'decoder_silhouette': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'h_shared': [],
            'h_depth_front': [],
            'h_depth_back': [],
            'h_silhouette': [],
        }
    else:
        # Initialize network.
        models = {
            'encoder': encoder.ResNetEncoder(torchvision.models.resnet.Bottleneck, [2, 3, 2], input_channel=3 if is_rgb else 2),
            'decoder_depth': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'decoder_silhouette': decoder.SingleBranchDecoder(in_size=2048, out_channels=1),
            'h_shared': [],
            'h_depth_front': [],
            'h_depth_back': [],
            'h_silhouette': [],
        }

    # output of model is
    # ((b, 1, 128, 128), (b, 1, 128, 128))

    for i in range(num_views // 2):
        models['h_shared'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
        models['h_depth_front'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
        models['h_depth_back'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))
        models['h_silhouette'].append(nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.ReLU(inplace=True)))

    sigmoid = nn.Sigmoid()

    losses = {
        'classification': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),  # for depth
        'bce': nn.BCELoss(),  # for silhouette
    }

    all_params = []

    def recursive_param_adder(module):
        if isinstance(module, (list, tuple)):
            for item in module:
                recursive_param_adder(item)
        elif isinstance(module, dict):
            for item in module.values():
                recursive_param_adder(item)
        elif isinstance(module, nn.Module):
            module.cuda()
            all_params.extend(module.parameters())
        else:
            raise RuntimeError('unknown module {}'.format(module))

    recursive_param_adder(models)
    assert len(all_params) > 0
    optimizer = optim.Adam(all_params, lr=learning_rate)

    print('model is ready.')


    # Initialize dataset.
    #####################################
    # loader = dataset.ExampleLoader('/data/mvshape/out/splits/shrec12_examples_vpo/train.bin', ['single_depth', 'multiview_depth', ], batch_size=batch_size, shuffle=True)

    # todo
    # loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_vpo/subset_examples.cbor'), ['input_rgb', 'target_depth', ], batch_size=batch_size, shuffle=True)
    loader = dataset.ExampleLoader2(path.join(dataset.base, 'out/splits/shapenetcore_examples_vpo/all_examples.cbor'), ['input_rgb', 'target_depth', ], batch_size=batch_size, shuffle=True)

    # eval_loader = dataset.ExampleLoader('/data/mvshape/out/splits/shrec12_examples_vpo/test.bin', ['single_depth', 'multiview_depth', ], batch_size=batch_size, subset_tags=['NOVELVIEW', 'NOVELMODEL'], shuffle=False)

    # Total number of examples.
    total_num_examples = loader.total_num_examples
    print('training dataset size: {}'.format(total_num_examples))
    #####################################


    data_queue = queue.Queue(maxsize=1)

    def prepare_data(next_batch):
        """
        :param next_batch: Output from data loader.
        :return: A dict of (field -> numpy array)
        """

        # A list of Example protobufs and a dict of tensors.
        examples, tensors = next_batch

        # prepare data
        # modify in-place.
        if is_rgb:
            in_rgb = tensors['input_rgb'].astype(np.float32) / 255.0
            in_image = in_rgb
        else:
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

    def convert_data_to_cuda(data_dict: dict):
        assert isinstance(data_dict, dict)
        return {key: torch.autograd.Variable(torch.from_numpy(value).cuda()) for key, value in data_dict.items()}

    def compute_classification_accuracy(out, classes):
        predicted = out.max(dim=1)[1]
        num_correct = (predicted == classes).int().sum().cpu().data.numpy()[0]
        num_total = predicted.size(0)
        accuracy = float(num_correct) / num_total
        return accuracy, num_correct, num_total

    def compute_masked_depth_loss(out_depth: Variable, target_depth: Variable, target_silhouette: Variable):
        target_depth_with_offset = (target_depth + target_depth_offset) * target_silhouette
        area = target_silhouette.float().sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) + 1e-3

        assert len(out_depth.size()) == 4
        assert len(target_depth.size()) == 4
        assert len(target_silhouette.size()) == 4

        masked_sq_sum = (torch.pow(out_depth - target_depth_with_offset, 2.0) * target_silhouette).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        out = (masked_sq_sum / area).mean()
        return out

    def compute_silhouette_loss(out_silhouette: Variable, target_silhouette: Variable):
        # area outside silhouette should be zero'ed out.
        out = losses['bce'](sigmoid(out_silhouette), target_silhouette)
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

    def data_queue_worker(data_loader: dataset.ExampleLoader, is_training):
        while True:
            next_batch = data_loader.next()
            if next_batch is None or (is_training and (len(next_batch[0]) != batch_size)):
                # if (data_loader.current_batch_index > 3) or next_batch is None or (is_training and (len(next_batch[0]) != batch_size)):
                print('{}: end of epoch {}'.format('Train' if is_training else 'Eval', current_epoch))
                data_loader.reset()
                data_queue.put(None, block=True)
                break

            # prepare
            batch_data_np = prepare_data(next_batch)
            # move to gpu
            batch_data = convert_data_to_cuda(batch_data_np)
            data_queue.put(batch_data, block=True)
        # print('end of data queue')
        gc.collect()

    def process_epoch(data_loader, is_training: bool):
        recursive_train_setter(models, is_training)
        gc.collect()

        all_outputs = {'out_depth_front': [], 'out_depth_back': [], 'out_silhouette': []}
        total_loss_sum = 0.0
        current_num_examples = 0
        current_num_batches = 0
        this_dataset_total_num_examples = data_loader.total_num_examples

        eval_metrics = {'silhouette_iou': 0.0}

        thread = threading.Thread(target=data_queue_worker, args=[data_loader, is_training], daemon=True)
        thread.start()

        while True:
            start_time = time.time()
            batch_data = data_queue.get(block=True)

            # Check if end of epoch.
            if batch_data is None:
                if is_training:
                    filename = path.join(dataset.base, 'out/pytorch/rgb_mv6_vpo/models0_{:04}.pth'.format(current_epoch))
                    io_utils.ensure_dir_exists(path.dirname(filename))
                    with open(filename, 'wb') as f:
                        torch.save(models, f)
                        print('saved {}'.format(filename))
                else:
                    if current_epoch > 0 and current_epoch % 5 == 0:
                        print('saving..')
                        with open('/tmp/eval.npz', 'wb') as f:
                            concatenated_front = np.concatenate(all_outputs['out_depth_front'], axis=0)
                            concatenated_back = np.concatenate(all_outputs['out_depth_back'], axis=0)
                            concatenated_silhouette = np.concatenate(all_outputs['out_silhouette'], axis=0)

                            ## important
                            ## TODO
                            # concatenated_front -= shrec12_target_depth_offset
                            # concatenated_back -= shrec12_target_depth_offset

                            np.savez_compressed(f, out_front=concatenated_front, out_back=concatenated_back, out_silhouette=concatenated_silhouette)
                            print('saved /tmp/eval.npz')
                break

            optimizer.zero_grad()

            # shared by all views
            model_input_image = batch_data['in_image']
            h = models['encoder'](model_input_image)

            total_loss = None
            for i in range(num_views // 2):
                # shared by front and back
                h_i = models['h_shared'][i](h)
                h_i_df = models['h_depth_front'][i](h_i)
                h_i_db = models['h_depth_back'][i](h_i)
                h_i_s = models['h_silhouette'][i](h_i)

                ## output images
                out_depth_front = models['decoder_depth'](h_i_df)  # (b, 1, 128, 128)
                out_depth_back = models['decoder_depth'](h_i_db)  # (b, 1, 128, 128)
                out_silhouette = models['decoder_silhouette'](h_i_s)  # (b, 1, 128, 128)

                ## target images
                target_depth_front = batch_data['target_depth'][:, i * 2]  # (b, 1, 128, 128)
                target_depth_back = batch_data['target_depth'][:, i * 2 + 1]  # (b, 1, 128, 128) already flipped
                target_silhouette = batch_data['target_silhouette'][:, i]  # (b, 1, 128, 128)

                # compute losses
                depth_loss_i_front = compute_masked_depth_loss(out_depth_front, target_depth_front, target_silhouette)
                depth_loss_i_back = compute_masked_depth_loss(out_depth_back, target_depth_back, target_silhouette)
                depth_loss_i = (depth_loss_i_front + depth_loss_i_back) * 0.5

                silhouette_loss_i = compute_silhouette_loss(out_silhouette, target_silhouette)
                total_loss_i = depth_loss_i + silhouette_loss_weight * silhouette_loss_i

                if total_loss is None:
                    total_loss = total_loss_i
                else:
                    total_loss += total_loss_i

                # silhouette iou
                silhouette_iou_i = compute_silhouette_iou(out_silhouette, target_silhouette=target_silhouette)
                eval_metrics['silhouette_iou'] += to_python_scalar(silhouette_iou_i) / (num_views // 2) * batch_data['in_image'].size(0) / this_dataset_total_num_examples

                if not is_training:
                    all_outputs['out_depth_front'].append(to_numpy(out_depth_front))
                    all_outputs['out_depth_back'].append(to_numpy(out_depth_back))
                    all_outputs['out_silhouette'].append(to_numpy(out_silhouette))

            total_loss /= num_views // 2

            # calling `.backward()` seems to trigger garbage collection. without it, it runs out of memory during testing.
            # this doesn't update any parameters, so it's safe to run when `is_training` is false.
            total_loss.backward()
            if is_training:
                optimizer.step()

            # running `to_python_scalar` triggers a sync point.
            total_loss_sum += (to_python_scalar(total_loss) * batch_data['in_image'].size(0)) / this_dataset_total_num_examples
            current_num_examples += batch_data['in_image'].size(0)
            current_num_batches += 1
            print('.', end='', flush=True)
            benchmark.print_elapsed()

        print('\n{} {} examples'.format('trained' if is_training else 'evaluated', current_num_examples))
        return total_loss_sum, eval_metrics

    # Main loop
    while True:
        print('# Epoch {}'.format(current_epoch))

        train_loss, eval_metrics = process_epoch(loader, is_training=True)
        print('Train loss: {:.3}, iou: {:.3}'.format(train_loss, eval_metrics['silhouette_iou']))

        # eval_loss, eval_metrics = process_epoch(eval_loader, is_training=False)
        # print('Eval loss: {:.3}, iou: {:.3}'.format(eval_loss, eval_metrics['silhouette_iou']))

        current_epoch += 1


if __name__ == '__main__':
    main()
