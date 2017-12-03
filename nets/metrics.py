import tensorflow as tf


def log_loss(logits, labels, reuse, num_examples, l2_weight):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = num_examples * tf.reduce_mean(cross_entropy, name='cross_entropy')
    l2_loss = 0
    if len(tf.get_collection('l2_loss')) > 0:
        l2_loss = tf.add_n(tf.get_collection('l2_loss'))
    total_loss = cross_entropy_mean + l2_weight * l2_loss
    if not reuse:
        tf.summary.scalar('l2_loss', l2_weight * l2_loss)
        tf.summary.scalar('nll_loss', cross_entropy_mean)
    return total_loss


def ssl_loss(logits, labels, reuse, num_examples, l2_weight, group_weight):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = num_examples * tf.reduce_mean(cross_entropy, name='cross_entropy')
    l2_loss = 0
    if len(tf.get_collection('l2_loss')) > 0:
        l2_loss = tf.add_n(tf.get_collection('l2_loss'))
    group_loss = 0
    for loss in tf.get_collection('filter_wise_loss'):
        group_loss += loss
    for loss in tf.get_collection('channel_wise_loss'):
        group_loss += loss
    total_loss = cross_entropy_mean + l2_weight * l2_loss + group_weight * group_loss
    if not reuse:
        tf.summary.scalar('l2_loss', l2_weight * l2_loss)
        tf.summary.scalar('group_loss', group_weight * group_loss)
        tf.summary.scalar('nll_loss', cross_entropy_mean)
    return total_loss


def sgvlb(logits, labels, reuse, num_examples, l2_weight, kl_weight=1):
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = num_examples * tf.reduce_mean(cross_entropy, name='cross_entropy')
    kl_loss = tf.add_n(tf.get_collection('kl_loss'))
    l2_loss = 0
    if len(tf.get_collection('l2_loss')) > 0:
        l2_loss = tf.add_n(tf.get_collection('l2_loss'))
    total_loss = cross_entropy_mean + kl_weight * kl_loss + l2_weight * l2_loss
    if not reuse:
        tf.summary.scalar('kl_loss', kl_weight * kl_loss)
        tf.summary.scalar('l2_loss', l2_weight * l2_loss)
        tf.summary.scalar('nll_loss', cross_entropy_mean)
    return total_loss


def accuracy(logits, labels):
    predicted_labels = tf.cast(tf.arg_max(logits, dimension=1), dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))
