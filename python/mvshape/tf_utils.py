import numpy as np
import tensorflow as tf


def logistic_loss(value: tf.Tensor, target: tf.Tensor, name='logistic_loss') -> tf.Tensor:
    """
    :param value: Output of the previous layer. Learned values are symmetric about the origin.
    e.g. [-1, 1]. When converting to binary labels, the threshold should be 0, rather than 0.5.
    :param target: Target Tensor with values in [0, 1].
    :param name: Name of the output Tensor.
    :return: A scalar float Tensor.
    """

    shape = value.get_shape().as_list()  # e.g. (None, 20, 64, 64)

    with tf.name_scope(name) as scope:
        if target.dtype == tf.bool:
            target = tf.cast(target, tf.float32, name='target_float')

        assert shape[-2] == shape[-3]  # Assume square image.
        assert target.dtype == value.dtype

        out = tf.nn.sigmoid_cross_entropy_with_logits(logits=value, labels=target, name='logistic_loss')
        loss = tf.reduce_mean(out, name=scope)
        return loss


def masked_l2(value: tf.Tensor, target: tf.Tensor, mask: tf.Tensor, name='masked_mse') -> tf.Tensor:
    """
    :param value: Output of the previous layer.
    :param target: Target Tensor.
    :param mask: Tensor with values in [0, 1]. 1 where object is visible. 0 where empty.
    :param name: Name of the output Tensor.
    :return: A scalar float Tensor.
    """
    shape = value.get_shape().as_list()  # e.g. (None, 20, 64, 64, 1)
    assert shape[-2] == shape[-3]  # Assume square image.
    assert mask.dtype == value.dtype

    with tf.name_scope(name) as scope:
        area = tf.add(tf.to_float(tf.reduce_sum(mask, reduction_indices=[2, 3], keep_dims=True, name='area')), 1e-3)
        masked_sq_sum = tf.reduce_sum(tf.multiply(tf.squared_difference(value, target), mask), reduction_indices=[2, 3], keep_dims=True)
        loss = tf.reduce_mean(tf.truediv(masked_sq_sum, area, name='weighted_mean'), name=scope)
    return loss


def conv_reshape(value,
                 k: int,
                 num_channels=None,
                 name: str = 'conv_reshape',
                 dims: int = 2) -> tf.Tensor:
    assert 2 <= dims <= 3

    shape = value.get_shape().as_list()
    assert len(shape) == 2, shape

    if num_channels is None:
        num_channels = float(shape[1]) / (k ** dims)
        assert num_channels.is_integer()
        num_channels = int(num_channels)
    out_shape = [-1] + ([k] * dims) + [num_channels]
    assert np.prod(out_shape[1:]) == np.prod(shape[1:])

    return tf.reshape(value, out_shape, name=name)


def shared_weight_op(values, func, scope_name):
    v = len(values)
    with tf.name_scope(scope_name):
        with tf.name_scope('stack_batch_dim'):
            h_concat = tf.concat(values=values, axis=1)  # [b, v, 4, 4, c]
            h_concat = tf.reshape(h_concat, [-1] + h_concat.get_shape().as_list()[2:])  # [b*v, 4, 4, c]

    out = func(h_concat)

    with tf.name_scope(scope_name):
        with tf.name_scope('expand_batch_dim'):
            if isinstance(out, (list, tuple)):
                ret = []
                for item in out:
                    out = tf.reshape(item, [-1, v] + item.get_shape().as_list()[1:])  # [b, v, 128, 128, c]
                    ret.append(out)
            else:
                out = tf.reshape(out, [-1, v] + out.get_shape().as_list()[1:])  # [b, v, 128, 128, c]
                ret = out

    return ret


def lrelu(x, alpha=0.1, name="lrelu"):
    i_scope = ""
    if hasattr(x, 'scope'):
        if x.scope: i_scope = x.scope
    with tf.name_scope(i_scope + name) as scope:
        x = tf.nn.relu(x)
        m_x = tf.nn.relu(-x)
        x -= alpha * m_x

    x.scope = scope

    return x


def mean_iou_2d(value: tf.Tensor, target: tf.Tensor, threshold=0.0, name='iou') -> tf.Tensor:
    """
    Input shape e.g. [None, 10, 128, 128, 1]
    """
    with tf.name_scope(name) as scope:
        v = tf.greater(value, threshold, name='thresholded')
        assert target.dtype == tf.bool
        dims = np.arange(target.get_shape().ndims - 3, target.get_shape().ndims)
        intersection = tf.reduce_sum(tf.to_float(tf.logical_and(v, target)), reduction_indices=dims)
        union = tf.reduce_sum(tf.to_float(tf.logical_or(v, target)), reduction_indices=dims)
        iou = tf.reduce_mean(tf.div(intersection, union), name=scope)
    return iou
