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
tf.app.flags.DEFINE_integer('batch_size', 100, 'batch size')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num GPUs')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_float('scale', 1.0, 'scale of width')
FLAGS = tf.app.flags.FLAGS


def conv_bn_rectify(net, num_filters, name, is_training, reuse, kl_weight=1.0):
    with tf.variable_scope(name):
        net = layers.conv_2d_layer(net, [3,3, net.get_shape()[3], num_filters], nonlinearity=None,
                                   padding='SAME', name='conv', with_biases=False, reuse=reuse)
        net = layers.sbp_dropout(net, 3, is_training, 'sbp', reuse, kl_weight=kl_weight)
        biases = layers.variable_on_cpu('biases', net.get_shape()[3], tf.constant_initializer(0.0), dtype=tf.float32)
        net = tf.nn.bias_add(net, biases)
        net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=False,
                                           center=True, scale=True)
        net = tf.nn.relu(net)
    return net


def net_vgglike(images, nclass, scale, is_training, reuse):
    net = conv_bn_rectify(images, int(64*scale), 'conv_1', is_training, reuse, 10.0)
    net = conv_bn_rectify(net, int(64*scale), 'conv_2', is_training, reuse, 10.0)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(128*scale), 'conv_3', is_training, reuse, 5.0)
    net = conv_bn_rectify(net, int(128*scale), 'conv_4', is_training, reuse, 5.0)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(256*scale), 'conv_5', is_training, reuse)
    net = conv_bn_rectify(net, int(256*scale), 'conv_6', is_training, reuse)
    net = conv_bn_rectify(net, int(256*scale), 'conv_7', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), 'conv_8', is_training, reuse)
    net = conv_bn_rectify(net, int(512*scale), 'conv_9', is_training, reuse)
    net = conv_bn_rectify(net, int(512*scale), 'conv_10', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = conv_bn_rectify(net, int(512*scale), 'conv_11', is_training, reuse)
    net = conv_bn_rectify(net, int(512*scale), 'conv_12', is_training, reuse)
    net = conv_bn_rectify(net, int(512*scale), 'conv_13', is_training, reuse)
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])
    net = layers.sbp_dropout(net, 1, is_training, 'sbp_dense_1', reuse)
    net = layers.dense_layer(net, net.get_shape()[1], int(512*scale),
                             nonlinearity=None, name='dense_1', with_biases=False)
    biases = layers.variable_on_cpu('biases_dense_1', net.get_shape()[1], tf.constant_initializer(0.0),
                                    dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    net = tf.contrib.layers.batch_norm(net, scope=tf.get_variable_scope(), reuse=reuse, is_training=False,
                                       center=True, scale=True)
    net = tf.nn.relu(net)
    net = layers.sbp_dropout(net, 1, is_training, 'sbp_dense_2', reuse)
    net = layers.dense_layer(net, net.get_shape()[1], nclass,
                             nonlinearity=None, name='dense_2', with_biases=False)
    biases = layers.variable_on_cpu('biases_dense_2', net.get_shape()[1], tf.constant_initializer(0.0),
                                    dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    return net


def main(_):
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/vgglike_sbp_{}_scale{}'.format(FLAGS.dataset, FLAGS.scale)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/vgglike_sbp_{}_scale{}'.format(FLAGS.dataset, FLAGS.scale)
        checkpoints_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')

    checkpoint_path = FLAGS.checkpoint
    batch_size = FLAGS.batch_size
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # DATASET QUEUES
        inputs, shape, n_train_examples, nclass = reader.get_producer(FLAGS.dataset, batch_size, training=True,
                                                                      distorted=True)
        images_train, labels_train = inputs
        inputs, shape, n_test_examples, nclass = reader.get_producer(FLAGS.dataset, batch_size, training=False)
        images_test, labels_test = inputs

        # BUILDING GRAPH
        devices = ['/gpu:%d' % i for i in range(FLAGS.num_gpus)]
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        wd = tf.placeholder(tf.float32, shape=[], name='weight_decay')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        inference = lambda images, is_training, reuse: net_vgglike(images, nclass, FLAGS.scale, is_training, reuse)
        loss = lambda preds, labels, reuse: metrics.sgvlb(preds, labels, reuse, n_train_examples, l2_weight=wd)
        operations = utils.build_graph_sbp(images_train, labels_train, images_test, labels_test, global_step, loss,
                                           metrics.accuracy, inference, lr, 10.0, devices)
        train_op, test_acc_op, test_loss_op = operations
        train_summaries = tf.summary.merge_all()
        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test_loss', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        n_epochs = 300
        steps_per_epoch = n_train_examples/(batch_size*FLAGS.num_gpus)+1
        steps_per_test = n_test_examples/(batch_size*FLAGS.num_gpus)+1
        lr_policy = lambda epoch_num: policies.linear_decay(epoch_num, decay_start=250, total_epochs=n_epochs,
                                                            start_value=1e-5)
        wd_policy = lambda epoch_num: 0.0

        checkpoint_saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            # restoring
            variables = filter(lambda v: 'adam' not in v.name.lower(), tf.get_collection('variables'))
            variables = filter(lambda v: 'beta1_power_1' not in v.name.lower(), variables)
            variables = filter(lambda v: 'beta2_power_1' not in v.name.lower(), variables)
            net_variables = filter(lambda v: 'sbp' not in v.name.lower(), variables)
            try:
                saver = tf.train.Saver(variables)
                saver.restore(sess, checkpoint_path)
            except tf.errors.NotFoundError as e:
                print 'variational variables are not found\nrestoring only net variables'
                saver = tf.train.Saver(net_variables)
                saver.restore(sess, checkpoint_path)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            best_test_acc = 0.0
            for epoch_num in range(n_epochs):
                for step_num in range(steps_per_epoch):
                    _, summary = sess.run([train_op, train_summaries], feed_dict={lr: lr_policy(epoch_num),
                                                                                  wd: wd_policy(epoch_num)})
                    train_writer.add_summary(summary, global_step.eval())
                test_loss_total, test_acc_total = 0.0, 0.0
                for step_num in range(steps_per_test):
                    batch_test_acc, batch_test_loss = sess.run([test_acc_op, test_loss_op])
                    test_acc_total += batch_test_acc/steps_per_test
                    test_loss_total += batch_test_loss/steps_per_test
                if test_acc_total >= best_test_acc:
                    checkpoint_saver.save(sess, checkpoints_dir + '/best_model.ckpt')
                    best_test_acc = test_acc_total
                checkpoint_saver.save(sess, checkpoints_dir + '/cur_model.ckpt')
                summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
                for s in summary:
                    test_writer.add_summary(s, global_step.eval())
            coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    tf.app.run()
