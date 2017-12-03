import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

from tensorflow.python.ops.distributions import special_math


def variable_on_cpu(name, shape, initializer, dtype, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype, trainable=trainable)
    return var


def add_l2_loss(var):
    tf.add_to_collection('l2_loss', tf.nn.l2_loss(var))


def add_filter_wise_loss(kernel, reuse):
    loss = tf.sqrt(tf.reduce_sum(kernel**2, [0, 1, 2])+1e-16)
    tf.add_to_collection('filter_wise_loss', tf.reduce_sum(loss))
    if not reuse:
        mask = tf.cast(tf.less(loss, 1e-5*tf.ones_like(loss)), tf.float32)
        tf.summary.scalar('filter_wise_sparsity', tf.reduce_sum(mask))


def add_channel_wise_loss(kernel, reuse):
    loss = tf.sqrt(tf.reduce_sum(kernel**2, [0, 1, 3])+1e-16)
    tf.add_to_collection('channel_wise_loss', tf.reduce_sum(loss))
    if not reuse:
        mask = tf.cast(tf.less(loss, 1e-5*tf.ones_like(loss)), tf.float32)
        tf.summary.scalar('channel_wise_sparsity', tf.reduce_sum(mask))


def dense_layer(input_tensor, num_inputs, num_outputs, nonlinearity, name, with_biases=True):
    with tf.variable_scope(name) as scope:
        W = variable_on_cpu('W',
                            [num_inputs, num_outputs],
                            tf.truncated_normal_initializer(stddev=1e-2, seed=322),
                            dtype=tf.float32)
        add_l2_loss(W)
        output = tf.matmul(input_tensor, W)
        if with_biases:
            biases = variable_on_cpu('biases', [num_outputs], tf.constant_initializer(0.0), dtype=tf.float32)
            output = tf.nn.bias_add(output, biases)
        if nonlinearity:
            output = nonlinearity(output, name=scope.name)
    return output


def conv_2d_layer(input_tensor, kernel_shape, nonlinearity, padding, name, strides=(1,1,1,1), with_biases=True,
                  reuse=False):
    """
    kernel_shape = [H, W, input_channels, num_filters]
    """
    with tf.variable_scope(name) as scope:
        kernel = variable_on_cpu('kernel',
                                 kernel_shape,
                                 tf.contrib.layers.xavier_initializer(seed=322),
                                 dtype=tf.float32)
        add_l2_loss(kernel)
        output = tf.nn.conv2d(input_tensor, kernel, strides, padding=padding)
        if with_biases:
            biases = variable_on_cpu('biases', kernel_shape[-1], tf.constant_initializer(0.0), dtype=tf.float32)
            output = tf.nn.bias_add(output, biases)
        if nonlinearity:
            output = nonlinearity(output, name=scope.name)
    return output


def phi(x):
    return 0.5*tf.erfc(-x/tf.sqrt(2.0))


def __erfinv(x):
    w = -tf.log((1.0-x)*(1.0+x)-1e-5)
    p_small = 2.81022636e-08*tf.ones_like(x)
    p_small = 3.43273939e-07 + p_small*(w-2.5)
    p_small = -3.5233877e-06 + p_small*(w-2.5)
    p_small = -4.39150654e-06 + p_small*(w-2.5)
    p_small = 0.00021858087 + p_small*(w-2.5)
    p_small = -0.00125372503 + p_small*(w-2.5)
    p_small = -0.00417768164 + p_small*(w-2.5)
    p_small = 0.246640727 + p_small*(w-2.5)
    p_small = 1.50140941 + p_small*(w-2.5)

    p_big = -0.000200214257*tf.ones_like(x)
    p_big = 0.000100950558 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00134934322 + p_big*(tf.sqrt(w) - 3.0)
    p_big = -0.00367342844 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00573950773 + p_big*(tf.sqrt(w) - 3.0)
    p_big = -0.0076224613 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 0.00943887047 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 1.00167406 + p_big*(tf.sqrt(w) - 3.0)
    p_big = 2.83297682 + p_big*(tf.sqrt(w) - 3.0)

    small_mask = tf.cast(tf.less(w, 5.0*tf.ones_like(w)), tf.float32)
    big_mask = tf.cast(tf.greater_equal(w, 5.0*tf.ones_like(w)), tf.float32)
    p = p_small*small_mask + p_big*big_mask
    return p*x


def erfinv(x):
    return special_math.ndtri((x+1.)/2.0)/tf.sqrt(2.)


def erfcx(x):
    """M. M. Shepherd and J. G. Laframboise,
       MATHEMATICS OF COMPUTATION 36, 249 (1981)
    """
    K = 3.75
    y = (tf.abs(x)-K) / (tf.abs(x)+K)
    y2 = 2.0*y
    (d, dd) = (-0.4e-20, 0.0)
    (d, dd) = (y2 * d - dd + 0.3e-20, d)
    (d, dd) = (y2 * d - dd + 0.97e-19, d)
    (d, dd) = (y2 * d - dd + 0.27e-19, d)
    (d, dd) = (y2 * d - dd + -0.2187e-17, d)
    (d, dd) = (y2 * d - dd + -0.2237e-17, d)
    (d, dd) = (y2 * d - dd + 0.50681e-16, d)
    (d, dd) = (y2 * d - dd + 0.74182e-16, d)
    (d, dd) = (y2 * d - dd + -0.1250795e-14, d)
    (d, dd) = (y2 * d - dd + -0.1864563e-14, d)
    (d, dd) = (y2 * d - dd + 0.33478119e-13, d)
    (d, dd) = (y2 * d - dd + 0.32525481e-13, d)
    (d, dd) = (y2 * d - dd + -0.965469675e-12, d)
    (d, dd) = (y2 * d - dd + 0.194558685e-12, d)
    (d, dd) = (y2 * d - dd + 0.28687950109e-10, d)
    (d, dd) = (y2 * d - dd + -0.63180883409e-10, d)
    (d, dd) = (y2 * d - dd + -0.775440020883e-09, d)
    (d, dd) = (y2 * d - dd + 0.4521959811218e-08, d)
    (d, dd) = (y2 * d - dd + 0.10764999465671e-07, d)
    (d, dd) = (y2 * d - dd + -0.218864010492344e-06, d)
    (d, dd) = (y2 * d - dd + 0.774038306619849e-06, d)
    (d, dd) = (y2 * d - dd + 0.4139027986073010e-05, d)
    (d, dd) = (y2 * d - dd + -0.69169733025012064e-04, d)
    (d, dd) = (y2 * d - dd + 0.490775836525808632e-03, d)
    (d, dd) = (y2 * d - dd + -0.2413163540417608191e-02, d)
    (d, dd) = (y2 * d - dd + 0.9074997670705265094e-02, d)
    (d, dd) = (y2 * d - dd + -0.26658668435305752277e-01, d)
    (d, dd) = (y2 * d - dd + 0.59209939998191890498e-01, d)
    (d, dd) = (y2 * d - dd + -0.84249133366517915584e-01, d)
    (d, dd) = (y2 * d - dd + -0.4590054580646477331e-02, d)
    d = y * d - dd + 0.1177578934567401754080e+01
    result = d/(1.0+2.0*tf.abs(x))
    result = tf.where(tf.is_nan(result), tf.ones_like(result), result)
    result = tf.where(tf.is_inf(result), tf.ones_like(result), result)

    negative_mask = tf.cast(tf.less(x, 0.0), tf.float32)
    positive_mask = tf.cast(tf.greater_equal(x, 0.0), tf.float32)
    negative_result = 2.0*tf.exp(x*x)-result
    negative_result = tf.where(tf.is_nan(negative_result), tf.ones_like(negative_result), negative_result)
    negative_result = tf.where(tf.is_inf(negative_result), tf.ones_like(negative_result), negative_result)
    result = negative_mask * negative_result + positive_mask * result
    return result


def phi_inv(x):
    return tf.sqrt(2.0)*erfinv(2.0*x-1)


def mean_truncated_log_normal_straight(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    mean = tf.exp(mu+sigma*sigma/2.0)/z*(phi(sigma-alpha) - phi(sigma-beta))
    return mean


def mean_truncated_log_normal_reduced(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    mean = erfcx((sigma-beta)/tf.sqrt(2.0))*tf.exp(b-beta*beta/2)
    mean = mean - erfcx((sigma-alpha)/tf.sqrt(2.0))*tf.exp(a-alpha*alpha/2)
    mean = mean/(2*z)
    return mean


def mean_truncated_log_normal(mu, sigma, a, b):
    return mean_truncated_log_normal_reduced(mu, sigma, a, b)


def median_truncated_log_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    gamma = phi(alpha)+0.5*(phi(beta)-phi(alpha))
    return tf.exp(phi_inv(gamma)*sigma+mu)


def snr_truncated_log_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    z = phi(beta) - phi(alpha)
    ratio = erfcx((sigma-beta)/tf.sqrt(2.0))*tf.exp((b-mu)-beta**2/2.0)
    ratio = ratio - erfcx((sigma-alpha)/tf.sqrt(2.0))*tf.exp((a-mu)-alpha**2/2.0)
    denominator = 2*z*erfcx((2.0*sigma-beta)/tf.sqrt(2.0))*tf.exp(2.0*(b-mu)-beta**2/2.0)
    denominator = denominator - 2*z*erfcx((2.0*sigma-alpha)/tf.sqrt(2.0))*tf.exp(2.0*(a-mu)-alpha**2/2.0)
    denominator = denominator - ratio**2
    ratio = ratio/tf.sqrt(denominator)
    return ratio


def sample_truncated_normal(mu, sigma, a, b):
    alpha = (a - mu)/sigma
    beta = (b - mu)/sigma
    gamma = phi(alpha)+tf.random_uniform(mu.shape)*(phi(beta)-phi(alpha))
    return tf.clip_by_value(phi_inv(tf.clip_by_value(gamma, 1e-5, 1.0-1e-5))*sigma+mu, a, b)


def sbp_dropout(input_tensor, axis, is_training, name, reuse=False, threshold=1.0, kl_weight=1.0):
    min_log = -20.0
    max_log = 0.0
    params_shape = np.ones(input_tensor.get_shape().ndims)
    params_shape[axis] = input_tensor.get_shape()[axis].value
    with tf.variable_scope(name) as scope:
        mu = variable_on_cpu('mu', dtype=tf.float32,
                             shape=params_shape.tolist(),
                             initializer=tf.zeros_initializer(), 
                             trainable=True)
        log_sigma = variable_on_cpu('log_sigma', dtype=tf.float32,
                                    shape=params_shape.tolist(),
                                    initializer=tf.constant_initializer(-5.0),
                                    trainable=True)
        log_sigma = tf.clip_by_value(log_sigma, -20.0, 5.0)
        mu = tf.clip_by_value(mu, -20.0, 5.0)
        sigma = tf.exp(log_sigma)

        # adding loss
        alpha = (min_log-mu)/sigma
        beta = (max_log-mu)/sigma
        z = phi(beta) - phi(alpha)

        def pdf(x):
            return tf.exp(-x*x/2.0)/tf.sqrt(2.0*np.pi)
        kl = -log_sigma-tf.log(z)-(alpha*pdf(alpha)-beta*pdf(beta))/(2.0*z)
        kl = kl+tf.log(max_log-min_log)-tf.log(2.0*np.pi*np.e)/2.0
        kl = kl_weight*tf.reduce_sum(kl)
        tf.add_to_collection('kl_loss', kl)

        # computing output
        if is_training:
            multiplicator = tf.exp(sample_truncated_normal(mu, sigma, min_log, max_log))
        else:
            multiplicator = mean_truncated_log_normal(mu, sigma, min_log, max_log)
        snr = snr_truncated_log_normal(mu, sigma, min_log, max_log)
        mask = tf.cast(tf.greater(snr, threshold*tf.ones_like(snr)), tf.float32)
        if not reuse:
            sparsity = tf.reduce_sum(mask)
            tf.summary.scalar('sparsity', sparsity)
            # tf.summary.histogram('mu', mu)
            # tf.summary.histogram('log_sigma', log_sigma)
            # tf.summary.histogram('mode', mode)
        if not is_training:
            multiplicator = mask*multiplicator
        output = multiplicator*input_tensor
    return output
