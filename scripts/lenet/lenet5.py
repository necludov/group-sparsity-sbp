import os
import sys
import time

if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())

import tensorflow as tf

from nets import layers, metrics, policies
from data import reader

tf.app.flags.DEFINE_string('dataset', 'mnist', 'dataset name')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_float('l2', 0.0, 'l2 regularizer coefficient')
FLAGS = tf.app.flags.FLAGS


def lenet5(images, nclass):
    # conv 1
    net = layers.conv_2d_layer(images, [5, 5, images.get_shape()[3], 20], nonlinearity=None,
                               padding='SAME', name='conv_1', with_biases=False)
    biases = layers.variable_on_cpu('biases_1', net.get_shape()[3], tf.constant_initializer(0.0), dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    net = tf.nn.relu(net)
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # conv 2
    net = layers.conv_2d_layer(net, [5, 5, net.get_shape()[3], 50], nonlinearity=None,
                               padding='SAME', name='conv_2', with_biases=False)
    biases = layers.variable_on_cpu('biases_2', net.get_shape()[3], tf.constant_initializer(0.0), dtype=tf.float32)
    net = tf.nn.bias_add(net, biases)
    net = tf.nn.relu(net)
    # max pool
    net = tf.nn.max_pool(net, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # reshape
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])
    # dense 1
    net = layers.dense_layer(net, net.get_shape()[1], 500,
                             nonlinearity=None, name='dense_1')
    net = tf.nn.relu(net)
    # dense 2
    net = layers.dense_layer(net, net.get_shape()[1], nclass,
                             nonlinearity=None, name='dense_2')
    return net


def main(_):
    batch_size = FLAGS.batch_size
    summaries_dir = FLAGS.summaries_dir
    if summaries_dir == '':
        summaries_dir = './logs/lenet5_{}_l2{}'.format(FLAGS.dataset, FLAGS.l2)
        summaries_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    checkpoints_dir = FLAGS.checkpoints_dir
    if checkpoints_dir == '':
        checkpoints_dir = './checkpoints/lenet5_{}_l2{}'.format(FLAGS.dataset, FLAGS.l2)
        checkpoints_dir += time.strftime('_%d-%m-%Y_%H:%M:%S')
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        # LOADING DATA
        data, len_train, len_test, input_shape, nclass = reader.load(FLAGS.dataset)
        X_train, y_train, X_test, y_test = data

        # BUILDING GRAPH
        images = tf.placeholder(tf.float32, shape=input_shape, name='images')
        labels = tf.placeholder(tf.int32, shape=[None], name='labels')
        lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        tf.summary.scalar('learning rate', lr)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.95)
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        logits_op = lenet5(images, nclass)
        loss_op = metrics.log_loss(logits_op, labels, reuse=False, num_examples=len_train, l2_weight=FLAGS.l2)
        accuracy_op = metrics.accuracy(logits_op, labels)
        tf.summary.scalar('train accuracy', accuracy_op)
        train_op = optimizer.minimize(loss_op, global_step=global_step)

        train_summaries = tf.summary.merge_all()
        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test accuracy', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test loss', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])

        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        n_epochs = 200
        lr_policy = lambda epoch_num: policies.linear_decay_policy(epoch_num, decay_start=100,
                                                                   total_epochs=n_epochs, start_value=1e-3)

        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # restore checkpoints if it's provided
            if FLAGS.checkpoint != '':
                restorer = tf.train.Saver(tf.get_collection('variables'))
                restorer.restore(sess, FLAGS.checkpoint)
            best_test_acc = 0.0
            for epoch_num in range(n_epochs):
                for i in range(len_train/batch_size+1):
                    batch_images, batch_labels = X_train[i*batch_size:(i+1)*batch_size], \
                                                 y_train[i*batch_size:(i+1)*batch_size]
                    _, summary = sess.run([train_op, train_summaries], feed_dict={lr: lr_policy(epoch_num),
                                                                                  images: batch_images,
                                                                                  labels: batch_labels})
                    train_writer.add_summary(summary, global_step.eval())
                test_loss_total, test_acc_total = 0.0, 0.0
                steps_per_test = len_test/batch_size+1
                for i in range(steps_per_test):
                    batch_images, batch_labels = X_test[i*batch_size:(i+1)*batch_size], \
                                                 y_test[i*batch_size:(i+1)*batch_size]
                    batch_test_acc, batch_test_loss = sess.run([accuracy_op, loss_op],
                                                               feed_dict={lr: lr_policy(epoch_num),
                                                                          images: batch_images,
                                                                          labels: batch_labels})
                    test_acc_total += batch_test_acc/steps_per_test
                    test_loss_total += batch_test_loss/steps_per_test
                if test_acc_total >= best_test_acc:
                    saver.save(sess, checkpoints_dir + '/best_model.ckpt')
                    best_test_acc = test_acc_total
                saver.save(sess, checkpoints_dir + '/cur_model.ckpt')
                summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
                for s in summary:
                    test_writer.add_summary(s, global_step.eval())

if __name__ == '__main__':
    tf.app.run()
