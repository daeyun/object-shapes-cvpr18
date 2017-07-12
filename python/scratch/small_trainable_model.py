import tensorflow as tf
import tensorflow.contrib.slim as slim
import mvshape.tf_utils
import shutil

save_dir = '/tmp/small_trainable_model'

""" Saver info:
filename tensor:  saver/Const:0
restore_op_name:  saver/restore_all
save_tensor_name:  saver/control_dependency:0
"""

""" Placeholders:
placeholder/images
placeholder/is_training
"""

""" Required variables and ops:
global_step
epoch

increment_global_step
increment_epoch

train_op
loss

saver/*
"""


def main():
    graph = tf.Graph()
    session = tf.Session(graph=graph)

    if mvshape.tf_utils.is_model_directory(save_dir):
        print('rm -r {}'.format(save_dir))
        shutil.rmtree(save_dir, ignore_errors=True)

    with graph.as_default():
        tf.set_random_seed(0)
        train_op, loss = model()

        vars_to_save = mvshape.tf_utils.variables_to_save(graph)
        saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=10, keep_checkpoint_every_n_hours=0.5, name='saver')
        saver_def = saver.as_saver_def()

        print('filename tensor: ', saver_def.filename_tensor_name)
        print('restore_op_name: ', saver_def.restore_op_name)
        print('save_tensor_name: ', saver_def.save_tensor_name)

        init_op = tf.global_variables_initializer()
        session.run(init_op)
        print('Initialized')

        builder = tf.saved_model.builder.SavedModelBuilder(save_dir)
        builder.add_meta_graph_and_variables(session, tags=["small_trainable_model"], clear_devices=True)
        builder.save(as_text=True)
        print('Saved', save_dir)


def model():
    with tf.variable_scope('placeholder'):
        images = tf.placeholder(tf.float32, (None, 64, 64, 1), name='images')
        is_training = tf.placeholder_with_default(False, (), 'is_training')

    # For debugging.
    tf.identity(is_training, name='identity/is_training')

    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={
                            'is_training': is_training, 'center': True, 'scale': True,
                            'fused': True, 'trainable': True, 'decay': 0.99}):
        value = slim.conv2d(images, 4, [3, 3], stride=2, scope='conv32')
        value = slim.conv2d(value, 8, [3, 3], stride=2, scope='conv16')
        value = slim.conv2d(value, 16, [3, 3], stride=2, scope='conv8')
        value = slim.conv2d(value, 32, [3, 3], stride=2, scope='conv4')
        value = slim.flatten(value, scope='flatten512')

        value = slim.fully_connected(value, 256, scope='fc256')
        value = tf.reshape(value, (-1, 4, 4, 16), name='reshape')
        value = slim.conv2d_transpose(value, 16, [3, 3], stride=2, scope='deconv8')
        value = slim.conv2d_transpose(value, 8, [3, 3], stride=2, scope='deconv16')
        value = slim.conv2d_transpose(value, 4, [3, 3], stride=2, scope='deconv32')
        value = slim.conv2d_transpose(value, 1, [3, 3], stride=2, scope='deconv64',
                                      activation_fn=None, biases_initializer=None, normalizer_fn=None)
        value = tf.identity(value, name='out')

        loss = tf.reduce_mean(tf.squared_difference(value, images), name='loss')

        with tf.device('/cpu:0'):
            global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int32),
                                          dtype=tf.int32, trainable=False)
            epoch = tf.get_variable('epoch', shape=(), initializer=tf.constant_initializer(0, dtype=tf.int32),
                                    dtype=tf.int32, trainable=False)
            increment_global_step = tf.assign_add(global_step, 1, use_locking=True, name='increment_global_step')
            increment_epoch = tf.assign_add(epoch, 1, use_locking=True, name='increment_epoch')

        optim = tf.train.AdamOptimizer(learning_rate=0.001, epsilon=1e-7, use_locking=False)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            minimize_op = optim.minimize(loss)
        train_op = tf.group(minimize_op, increment_global_step, name='train_op')

    return train_op, loss


if __name__ == '__main__':
    main()
