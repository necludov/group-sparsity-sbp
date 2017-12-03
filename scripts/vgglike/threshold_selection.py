import os
import sys
import time

if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from nets import layers, utils, metrics, policies
from data import reader

tf.app.flags.DEFINE_string('dataset', 'cifar10', 'dataset name')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_float('scale', 1.0, 'scale of width')
FLAGS = tf.app.flags.FLAGS


def conv_bn_rectify(net, num_filters, wd, name, is_training, reuse, threshold):
    with tf.variable_scope(name):
        net = layers.conv_2d_layer(net, [3,3], net.get_shape()[3], num_filters,
                                   nonlinearity=None, wd=wd, padding='SAME', name='conv', with_biases=False)
        net = layers.sbp_dropout(net, 3, is_training, 'sbp', reuse, threshold)
        biases = layers._variable_on_cpu('biases', net.get_shape()[3], tf.constant_initializer(0.0), dtype=tf.float32)
        net = tf.nn.bias_add(net, biases)
        net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=False,
                                           center=True, scale=True)
        net = tf.nn.relu(net)
    return net


def net_vgglike(images, nclass, scale, is_training, reuse, threshold):
    net = conv_bn_rectify(images, int(64*scale), 0.0, 'conv_1', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(64*scale), 0.0, 'conv_2', is_training, reuse, threshold)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(128*scale), 0.0, 'conv_3', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(128*scale), 0.0, 'conv_4', is_training, reuse, threshold)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(256*scale), 0.0, 'conv_5', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(256*scale), 0.0, 'conv_6', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(256*scale), 0.0, 'conv_7', is_training, reuse, threshold)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_8', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_9', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_10', is_training, reuse, threshold)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_11', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_12', is_training, reuse, threshold)
    net = conv_bn_rectify(net, int(512*scale), 0.0, 'conv_13', is_training, reuse, threshold)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])
    net = layers.sbp_dropout(net, 1, is_training, 'sbp_dense_1', reuse, threshold)
    net = layers.dense_layer(net, net.get_shape()[1], int(512*scale),
                             nonlinearity=None, wd=0.0, name='dense_1', with_biases=False)
    biases = layers._variable_on_cpu('biases_dense_1', net.get_shape()[1], tf.constant_initializer(0.0),
                                     dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=False,
                                       center=True, scale=True)
    net = tf.nn.relu(net)
    net = layers.sbp_dropout(net, 1, is_training, 'sbp_dense_2', reuse, threshold)
    net = layers.dense_layer(net, net.get_shape()[1], nclass,
                             nonlinearity=None, wd=0.0, name='dense_2', with_biases=False)
    biases = layers._variable_on_cpu('biases_dense_2', net.get_shape()[1], tf.constant_initializer(0.0),
                                     dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    return net


def main(_):
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/threshold_selection_sbp_{}_scale{}'.format(FLAGS.dataset, FLAGS.scale)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')

    checkpoint_path = FLAGS.checkpoint
    batch_size = FLAGS.batch_size
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # DATASET QUEUES
        inputs, shape, n_train_examples, nclass = reader.get_producer(FLAGS.dataset, batch_size, training=True)
        images_train, labels_train = inputs
        inputs, shape, n_test_examples, nclass = reader.get_producer(FLAGS.dataset, batch_size, training=False)
        images_test, labels_test = inputs
        train_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_train, labels_train])
        test_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images_test, labels_test])

        # BUILDING GRAPH
        threshold = tf.placeholder(tf.float32, shape=[], name='threshold')
        tf.summary.scalar('thershold', threshold)
        inference = lambda images, is_training, reuse: net_vgglike(images, nclass, FLAGS.scale, is_training,
                                                                   reuse, threshold)
        loss = lambda preds, labels, reuse: metrics.sgvlb(preds, labels, reuse, n_train_examples)
        with tf.variable_scope(tf.get_variable_scope()) as scope:
            with tf.name_scope('tower') as scope:
                # train ops
                batch_images_train, batch_labels_train = train_queue.dequeue()
                train_preds = inference(batch_images_train, is_training=True, reuse=False)
                train_acc_op = metrics.accuracy(train_preds, batch_labels_train)
                train_loss_op = loss(train_preds, batch_labels_train, reuse=False)
                tf.get_variable_scope().reuse_variables()

                # test ops
                batch_images_test, batch_labels_test = test_queue.dequeue()
                test_preds = inference(batch_images_test, is_training=False, reuse=True)
                test_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=test_preds,
                                                                              labels=batch_labels_test)
                test_loss_op = tf.reduce_mean(test_loss_op)
                test_acc_op = metrics.accuracy(test_preds, batch_labels_test)

        train_acc = tf.placeholder(tf.float32, shape=[], name='train_acc_placeholder')
        tf.summary.scalar('train_accuracy_threshold', train_acc)
        train_loss = tf.placeholder(tf.float32, shape=[], name='train_loss_placeholder')
        tf.summary.scalar('train_loss_threshold', train_loss)
        train_summaries = tf.summary.merge_all()

        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy_threshold', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test_loss_threshold', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        steps_per_train = n_train_examples/batch_size+1
        steps_per_test = n_test_examples/batch_size+1

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # restoring
            variables = filter(lambda v: 'adam' not in v.name.lower(), tf.get_collection('variables'))
            variables = filter(lambda v: 'beta1_power_1' not in v.name.lower(), variables)
            variables = filter(lambda v: 'beta2_power_1' not in v.name.lower(), variables)
            saver = tf.train.Saver(variables)
            saver.restore(sess, checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            iteration = 0
            for t in np.arange(0.0, 2.1, 1e-1):
                train_acc_total, train_loss_total = 0.0, 0.0
                for step_num in range(steps_per_train):
                    batch_train_acc, batch_train_loss = sess.run([train_acc_op, train_loss_op],
                                                                 feed_dict={threshold: t})
                    train_acc_total += batch_train_acc/steps_per_train
                    train_loss_total += batch_train_loss/steps_per_train
                summary = sess.run([train_summaries],
                                   feed_dict={train_acc: train_acc_total, train_loss: train_loss_total, threshold: t})
                for s in summary:
                    train_writer.add_summary(s, iteration)

                test_loss_total, test_acc_total = 0.0, 0.0
                for step_num in range(steps_per_test):
                    batch_test_acc, batch_test_loss = sess.run([test_acc_op, test_loss_op], feed_dict={threshold: t})
                    test_acc_total += batch_test_acc/steps_per_test
                    test_loss_total += batch_test_loss/steps_per_test

                summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
                for s in summary:
                    test_writer.add_summary(s, iteration)
                iteration += 1
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
