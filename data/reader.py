import os
import gzip
import numpy as np
import cPickle as pickle

import cifar10_input


def get_producer(dataset, batch_size, training, smaller_final_batch=False, distorted=False,
                 data_dir='./data/cifar-10-batches-bin/'):
    if dataset == 'cifar10':
        return cifar10_producer(training, batch_size, smaller_final_batch, distorted, data_dir)
    raise Exception('Load of %s not implimented yet' % dataset)


def cifar10_producer(training, batch_size, smaller_final_batch, distorted, data_dir):
    if training:
        if distorted:
            inputs = cifar10_input.distorted_inputs(data_dir=data_dir,
                                                    batch_size=batch_size,
                                                    smaller_final_batch=smaller_final_batch)
        else:
            inputs = cifar10_input.inputs(training=training,
                                          data_dir=data_dir,
                                          batch_size=batch_size,
                                          smaller_final_batch=smaller_final_batch)
    else:
        if distorted:
            raise Exception('distorted inputs are not available for test')
        inputs = cifar10_input.inputs(training=training,
                                      data_dir=data_dir,
                                      batch_size=batch_size,
                                      smaller_final_batch=smaller_final_batch)
    if smaller_final_batch:
        shape = [None, cifar10_input.IMAGE_SIZE, cifar10_input.IMAGE_SIZE, 3]
    else:
        shape = [batch_size, cifar10_input.IMAGE_SIZE, cifar10_input.IMAGE_SIZE, 3]
    if training:
        num_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        num_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
    return inputs, shape, num_examples, cifar10_input.NUM_CLASSES


def load(dataset):
    if dataset == 'mnist':
        return load_mnist()
    if dataset == 'mnist-random':
        return load_mnist_random()
    if dataset == 'cifar10':
        return load_cifar10()
    if dataset == 'cifar10-random':
        return load_cifar10_random()
    if dataset == 'cifar100':
        return load_cifar100()
    raise Exception('Load of %s not implemented yet' % dataset)


def load_mnist(base='./data/mnist'):
    """
    load_mnist taken from https://github.com/Lasagne/Lasagne/blob/master/examples/images.py
    :param base: base path to images dataset
    """

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28).transpose(0,2,3,1)
        return data / np.float32(255)

    def load_mnist_labels(filename):
        with gzip.open(filename, 'rb') as f:
            Y = np.frombuffer(f.read(), np.uint8, offset=8)
        return Y

    # We can now download and read the training and test set image and labels.
    X_train = load_mnist_images(base + '/train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels(base + '/train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images(base + '/t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels(base + '/t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 28, 28, 1), 10


def load_mnist_random(base='./data/mnist'):
    X_train, y_train, X_test, y_test = load_mnist(base)[0]
    np.random.seed(74632)
    y_train = np.random.choice(10, len(y_train))
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 28, 28, 1), 10


def load_cifar10(base='./data/cifar10'):
    def load_CIFAR_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            Y = np.array(datadict['labels'])
            X = datadict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            #X = X/255.0
            Y_oh = np.zeros([len(Y), 10])
            Y_oh[np.arange(len(Y)), Y] = 1
            return X, Y_oh

    def load_CIFAR10(ROOT):
        xs, ys = [], []
        for b in range(1, 6):
            f = os.path.join(ROOT, 'data_batch_%d' % (b,))
            X, Y = load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
        del X, Y
        Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    # Load the raw CIFAR-10 data
    cifar10_dir = os.path.join(base, 'cifar-10-batches-py')
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # mean_image = np.mean(X_train, axis=0)
    # X_train -= mean_image
    # X_test -= mean_image

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10


def load_cifar10_random(base='./data/cifar10'):
    X_train, y_train, X_test, y_test = load_cifar10(base)[0]
    np.random.seed(74632)
    y_train = np.random.choice(10, len(y_train))
    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 10


def load_cifar100(base='./data/cifar100/cifar-100-python/'):
    def load_CIFAR_batch(filename, num):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            X = datadict['data']
            Y = datadict['coarse_labels']
            X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            X = X/255.0
            Y = np.array(Y)
            Y_oh = np.zeros([len(Y), 100])
            Y_oh[np.arange(len(Y)), Y] = 1
            return X, Y_oh

    Xtr, Ytr = load_CIFAR_batch(os.path.join(base, 'train'), 50000)
    Xte, Yte = load_CIFAR_batch(os.path.join(base, 'test'), 10000)

    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    return (X_train, y_train, X_test, y_test), X_train.shape[0], X_test.shape[0], (None, 32, 32, 3), 100
