import numpy as np
import numpy.linalg as la
import tensorflow as tf
from tensorflow.contrib import slim
from mvshape import tf_utils
import shutil


def model():
    def _encoder(in_depth, in_silhouette):
        def _encoder_input(value, d=32, name='enc'):
            out = value
            with tf.variable_scope(name):
                out = slim.conv2d(out, d, [7, 7], stride=1, rate=3, scope='conv128.1')
                out = slim.conv2d(out, d, [5, 5], stride=2, rate=1, scope='conv64.1')
            return out

        enc_d = _encoder_input(in_depth, 24, name='depth_128_64')
        enc_s = _encoder_input(in_silhouette, 12, name='silhouette_128_64')

        out = tf.concat(values=[enc_d, enc_s], axis=3)
        out = slim.conv2d(out, 32, [3, 3], stride=1, scope='conv64.2')
        out = slim.conv2d(out, 32, [3, 3], stride=1, scope='conv64.3')
        out = slim.conv2d(out, 64, [3, 3], stride=2, scope='conv32.1')
        out = slim.conv2d(out, 64, [3, 3], stride=1, scope='conv32.2')
        out = slim.conv2d(out, 64, [3, 3], stride=1, scope='conv32.3')
        out = slim.conv2d(out, 96, [3, 3], stride=2, scope='conv16.1')
        out = slim.conv2d(out, 96, [3, 3], stride=1, scope='conv16.2')
        out = slim.conv2d(out, 96, [3, 3], stride=1, scope='conv16.3')
        out = slim.conv2d(out, 128, [3, 3], stride=2, scope='conv8.1')
        out = slim.conv2d(out, 128, [3, 3], stride=1, scope='conv8.2')
        out = slim.conv2d(out, 128, [3, 3], stride=1, scope='conv8.3')
        out = slim.conv2d(out, 512, [3, 3], stride=2, scope='conv4.1')
        out = slim.conv2d(out, 512, [4, 4], stride=2, padding='VALID', scope='fc')
        out = tf.squeeze(out, [1, 2], name='fc/squeezed')
        h_shared = slim.fully_connected(out, 1024, scope='fc2')

        return h_shared

    def _decoder(value, name):
        with tf.variable_scope(name):
            value = slim.conv2d_transpose(value, 256, [5, 5], stride=2, scope='deconv8.1')
            value = slim.conv2d_transpose(value, 128, [5, 5], stride=2, scope='deconv16.1')
            value = slim.conv2d_transpose(value, 64, [5, 5], stride=2, scope='deconv32.1')
            value = slim.conv2d_transpose(value, 32, [5, 5], stride=2, scope='deconv64.1')
            value = slim.conv2d_transpose(value, 1, [5, 5], activation_fn=None, biases_initializer=None, normalizer_fn=None, stride=2, scope='deconv128.1')
        return value

    def _decoder2(value, name):
        with tf.variable_scope(name):
            value = slim.conv2d_transpose(value, 256, [4, 4], stride=2, scope='deconv16.1')
            value = slim.conv2d_transpose(value, 256, [4, 4], stride=1, scope='deconv16.2')
            value = slim.conv2d_transpose(value, 92, [4, 4], stride=2, scope='deconv32.1')
            value = slim.conv2d_transpose(value, 92, [4, 4], stride=1, scope='deconv32.2')
            value = slim.conv2d_transpose(value, 48, [4, 4], stride=2, scope='deconv64.1')
            value = slim.conv2d_transpose(value, 48, [4, 4], stride=1, scope='deconv64.2')
            # TODO
            value1 = slim.conv2d_transpose(value, 1, [4, 4], activation_fn=None, biases_initializer=None, normalizer_fn=None, stride=2, scope='deconv128.1.1')
            value2 = slim.conv2d_transpose(value, 1, [4, 4], activation_fn=None, biases_initializer=None, normalizer_fn=None, stride=2, scope='deconv128.1.2')
        return value1, value2

    num_branches = 6

    # Placeholders.
    with tf.variable_scope('placeholder'):
        in_depth = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='in_depth')
        target_depth = tf.placeholder(dtype=tf.float32, shape=[None, num_branches, 128, 128, 1], name='target_depth')
        is_training = tf.placeholder_with_default(False, (), 'is_training')
        target_depth_offset = tf.placeholder_with_default(tf.constant(0.0, dtype=tf.float32, shape=()), (), 'target_depth_offset')

    in_depth_mask = tf.is_finite(in_depth, name='in_depth_mask')
    in_silhouette = tf.cast(in_depth_mask, tf.float32, name='in_silhouette')
    target_depth_mask = tf.is_finite(target_depth, name='target_depth_mask')
    target_silhouette = tf.cast(target_depth_mask, tf.float32, name='target_silhouette')

    target_depth = tf.where(target_depth_mask, target_depth + target_depth_offset,
                            tf.zeros_like(target_depth), name='target_depth_zeroed')
    in_depth = tf.where(in_depth_mask, in_depth, tf.zeros_like(in_depth), name='in_depth_zeroed')

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training,
                                           'center': True, 'scale': True,
                                           'fused': True, 'trainable': True, 'decay': 0.99},
                        outputs_collections='out'):
        h = _encoder(in_depth=in_depth, in_silhouette=in_silhouette)

        h_list = []
        for i in range(num_branches):
            with tf.variable_scope('view_{}'.format(i)):
                out = slim.fully_connected(h, 1024, scope='fc')
                out = tf_utils.conv_reshape(out, 8, dims=2, name='conv_reshape')
                out = tf.expand_dims(out, 1)
                h_list.append(out)

        out_d, out_s = tf_utils.shared_weight_op(h_list, lambda x: _decoder2(x, name='decoder'),
                                                 scope_name='weight_sharing')

        tf.identity(out_d, name='out_depth')
        tf.identity(out_s, name='out_silhouette')
        iou = tf_utils.mean_iou_2d(out_s, target_depth_mask, threshold=0.0, name='iou')

        loss_s = tf_utils.logistic_loss(out_s, target_silhouette, name='loss_s')
        loss_d = tf_utils.masked_l2(out_d, target_depth, target_silhouette, name='loss_d')

        silhouette_weight = 0.2

        loss = tf.identity(loss_s * silhouette_weight + loss_d, name='loss')

    with tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int32),
                                      dtype=tf.int32, trainable=False)
        epoch = tf.get_variable('epoch', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int32),
                                dtype=tf.int32, trainable=False)
        increment_global_step = tf.assign_add(global_step, 1, use_locking=True, name='increment_global_step')
        increment_epoch = tf.assign_add(epoch, 1, use_locking=True, name='increment_epoch')

    optim = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-7, use_locking=False)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        minimize_op = optim.minimize(loss)
    train_op = tf.group(minimize_op, increment_global_step, name='train_op')

    return train_op, loss


def main():
    config = tf.ConfigProto(
        device_count={'GPU': 1}
    )

    import os
    savedir = '/data/mvshape/out/tf_models/depth_6_views/'
    shutil.rmtree(savedir, ignore_errors=True)

    save = not os.path.isdir(savedir)
    print('save', save)

    graph = tf.Graph()

    with graph.as_default():
        tf.set_random_seed(0)
        train_op, loss = model()

        vars_to_save = graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        vars_to_save += graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        vars_to_save += graph.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
        unique_vars = {}
        for item in vars_to_save:
            unique_vars[item.name] = item
        vars_to_save = list(unique_vars.values())
        print(len(vars_to_save))

        saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=50, keep_checkpoint_every_n_hours=0.5, name='saver')
        saver_def = saver.as_saver_def()

        print(saver_def.filename_tensor_name)
        print(saver_def.restore_op_name)
        print(saver_def.save_tensor_name)

        init_op = tf.global_variables_initializer()

    sess = tf.Session(graph=graph, config=config)

    with graph.as_default():
        sess.run(init_op)

    if save:
        with graph.as_default():
            builder = tf.saved_model.builder.SavedModelBuilder(savedir)
            builder.add_meta_graph_and_variables(sess, ["mv"], clear_devices=True)
            builder.save()


if __name__ == '__main__':
    main()
