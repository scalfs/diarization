import tensorflow as tf


def normalize(x):
    """ normalize the last dimension vector of the input matrix
    :return: normalized input
    """
    return x/tf.sqrt(tf.reduce_sum(x**2, axis=-1, keep_dims=True)+1e-6)
