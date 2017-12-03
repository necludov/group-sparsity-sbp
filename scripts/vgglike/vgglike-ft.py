import os
import sys
import time

if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())

import tensorflow as tf

from nets import layers, utils, metrics, policies
from data import reader

tf.app.flags.DEFINE_string('dataset', 'cifar10', 'dataset name')
tf.app.flags.DEFINE_string('data_dir', './data/cifar-10-batches-bin/', 'Path to data')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_integer('batch_size', 25, 'batch size')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num GPUs')
FLAGS = tf.app.flags.FLAGS


def conv_bn_rectify(net, num_filters, name, is_training, reuse):
    with tf.variable_scope(name):
        net = layers.conv_2d_layer(net, [3, 3, net.get_shape()[3], num_filters], nonlinearity=None,
                                   padding='SAME', name='conv', with_biases=False, reuse=reuse)
        biases = layers.variable_on_cpu('biases', num_filters, tf.constant_initializer(0.0), dtype=tf.float32)
        net = tf.nn.bias_add(net, biases)
        net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=is_training,
                                           center=False, scale=True)
        net = tf.nn.relu(net)
    return net


def net_vgglike(images, nclass, num_filters, is_training, reuse):
    net = conv_bn_rectify(images, num_filters[0], 'conv_1', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.7, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[1], 'conv_2', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, num_filters[2], 'conv_3', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[3], 'conv_4', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, num_filters[4], 'conv_5', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[5], 'conv_6', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[6], 'conv_7', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, num_filters[7], 'conv_8', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[8], 'conv_9', is_training, reuse)
    net = tf.contrib.layers.dropout(net, keep_prob=0.6, is_training=is_training)
    net = conv_bn_rectify(net, num_filters[9], 'conv_10', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, num_filters[10], 'conv_11', is_training, reuse)
    net = conv_bn_rectify(net, num_filters[11], 'conv_12', is_training, reuse)
    net = conv_bn_rectify(net, num_filters[12], 'conv_13', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])

    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = layers.dense_layer(net, net.get_shape()[1], 512,
                             nonlinearity=None, name='dense_1', with_biases=False)
    biases = layers.variable_on_cpu('biases_dense_1', net.get_shape()[1], tf.constant_initializer(0.0),
                                    dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=is_training,
                                       center=True, scale=True)
    net = tf.nn.relu(net)
    net = tf.contrib.layers.dropout(net, keep_prob=0.5, is_training=is_training)
    net = layers.dense_layer(net, net.get_shape()[1], nclass,
                             nonlinearity=None, name='dense_2', with_biases=False)
    biases = layers.variable_on_cpu('biases_dense_2', net.get_shape()[1], tf.constant_initializer(0.0),
                                    dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    tf.add_to_collection('logits', net)
    return net


def main(_):
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/vgglike_ft_{}'.format(FLAGS.dataset)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/vgglike_ft_{}'.format(FLAGS.dataset)
        checkpoints_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # DATASET QUEUES
        inputs, shape, n_train_examples, nclass = reader.get_producer(FLAGS.dataset, FLAGS.batch_size, training=True,
                                                                      distorted=True, data_dir=FLAGS.data_dir)
        images_train, labels_train = inputs
        inputs, shape, n_test_examples, nclass = reader.get_producer(FLAGS.dataset, FLAGS.batch_size, training=False,
                                                                     data_dir=FLAGS.data_dir)
        images_test, labels_test = inputs

        # BUILDING GRAPH
        devices = ['/gpu:%d' % i for i in range(FLAGS.num_gpus)]
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        wd = tf.placeholder(tf.float32, shape=[], name='weight_decay')
        group_weight = tf.placeholder(tf.float32, shape=[], name='group_weight')
        tf.summary.scalar('weight_decay', wd)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_filters = [64, 64, 128, 128, 256, 256, 256, 509, 417, 126, 136, 83, 512]
        inference = lambda images, is_training, reuse: net_vgglike(images, nclass, num_filters, is_training, reuse)
        loss = lambda preds, labels, reuse: metrics.ssl_loss(preds, labels, reuse, n_train_examples, wd, group_weight)
        train_op, test_acc_op, test_loss_op = utils.build_graph(images_train, labels_train, images_test, labels_test,
                                                                global_step, loss, metrics.accuracy,
                                                                inference, lr, devices)
        train_summaries = tf.summary.merge_all()
        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test_loss', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)
        saver = tf.train.Saver()

    # TRAINING
    n_epochs = 200
    steps_per_epoch = n_train_examples/(FLAGS.batch_size*FLAGS.num_gpus)+1
    steps_per_test = n_test_examples/(FLAGS.batch_size*FLAGS.num_gpus)+1
    lr_policy = lambda epoch_num: policies.linear_decay(epoch_num, decay_start=100, total_epochs=n_epochs,
                                                        start_value=1e-4)
    wd_policy = lambda epoch_num: 1.0
    gw_policy = lambda epoch_num: 1.0

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    with tf.Session(config=config, graph=graph) as sess:
        # initialize all variables
        sess.run(tf.global_variables_initializer())

        # restore checkpoints if it's provided
        if FLAGS.checkpoint != '':
            variables_to_restore = filter(lambda v: 'adam' not in v.name.lower(), tf.get_collection('variables'))
            variables_to_restore = filter(lambda v: ('conv' in v.name.lower()) or ('dense' in v.name.lower()),
                                          variables_to_restore)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.checkpoint)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        best_test_acc = 0.0
        for epoch_num in range(1):
            for step_num in range(steps_per_epoch):
                _, summary = sess.run([train_op, train_summaries], feed_dict={lr: lr_policy(epoch_num),
                                                                              wd: wd_policy(epoch_num),
                                                                              group_weight: gw_policy(epoch_num)})
                train_writer.add_summary(summary, global_step.eval())
            test_loss_total, test_acc_total = 0.0, 0.0
            for step_num in range(steps_per_test):
                batch_test_acc, batch_test_loss = sess.run([test_acc_op, test_loss_op])
                test_acc_total += batch_test_acc/steps_per_test
                test_loss_total += batch_test_loss/steps_per_test
            if test_acc_total >= best_test_acc:
                saver.save(sess, checkpoints_dir + '/best_model.ckpt')
                best_test_acc = test_acc_total
            saver.save(sess, checkpoints_dir + '/cur_model.ckpt')
            summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
            for s in summary:
                test_writer.add_summary(s, global_step.eval())
            print("Epoch %d test accuracy: %.3f" % (epoch_num, test_acc_total))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
