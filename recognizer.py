#! /usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import imageio
import PIL
from PIL import Image


def load_data_set(base_path, test_train, one_hot=False):
    # See http://yann.lecun.com/exdb/mnist/
    if test_train == "train":
        image_path = base_path + "/train-images-idx3-ubyte.gz"
        label_path = base_path + "/train-labels-idx1-ubyte.gz"
    elif test_train == "test":
        image_path = base_path + "/t10k-images-idx3-ubyte.gz"
        label_path = base_path + "/t10k-labels-idx1-ubyte.gz"
    else:
        raise Exception("Only test set and train set, please pass \"test\" or \"train\"")

    with gzip.open(image_path, "r") as f:
        magic_number = f.read(4)  # 0x00000803
        image_number = f.read(4)  # 60000
        row_size = f.read(4)      # 28
        col_size = f.read(4)      # 28

        image_number = int.from_bytes(image_number, "big")
        row_size = int.from_bytes(row_size, "big")
        col_size = int.from_bytes(col_size, "big")

        buf = f.read(row_size * col_size * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        images = data.reshape(image_number, row_size * col_size)

    with gzip.open(label_path, "r") as f:
        magic_number = f.read(4)  # 0x00000801
        label_number = f.read(4)  # 60000

        label_number = int.from_bytes(label_number, "big")

        buf = f.read(label_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        labels = data
        if one_hot:
            labels = to_categorical(labels)

    return images, labels


def shuffle_samples(x, y):
    permutation = np.random.permutation(y.shape[0])
    shuffled_x = x[permutation, :]
    shuffled_y = y[permutation]

    return shuffled_x, shuffled_y


def get_next_batch(x, y, start, end):
    x_batch = x[start:end]
    y_batch = y[start:end]

    return x_batch, y_batch


def weight_variable(shape):
    initer = tf.truncated_normal_initializer(stddev=0.01)

    return tf.get_variable("W", dtype=tf.float32, shape=shape, initializer=initer)


def bias_variable(shape):
    initer = tf.constant(0.0, dtype=tf.float32, shape=shape)

    return tf.get_variable("b", dtype=tf.float32, initializer=initer)


def print_image(image):
    shape = image.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            dot = image[i][j]
            if dot == 0:
                print("  ", end="")
            else:
                print("##", end="")

        print("")


def train_model(train_x, train_y, valid_x, valid_y, path=".model", name="mnist_lr"):
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, 784], name="X")
    y = tf.placeholder(tf.float32, [None, 10], name="Y")

    W = weight_variable([784, 10])
    b = bias_variable([10])

    # Hyper-parameters
    epochs = 10             # Total number of training epochs
    batch_size = 100        # Training batch size
    display_freq = 100      # Frequency of displaying the training results
    learning_rate = 0.001   # The optimization initial learning rate

    output = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(y, output), name="loss")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    pred = tf.argmax(output, 1, name="pred")
    correct_prediction = tf.equal(pred, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        num_tr_iter = int(len(train_y) / batch_size)

        for epoch in range(epochs):
            print(f"train epoch {epoch+1}")
            epoch_x, epoch_y = shuffle_samples(train_x, train_y)
            for iteration in range(num_tr_iter):
                start = iteration * batch_size
                end = (iteration + 1) * batch_size
                batch_x, batch_y = get_next_batch(epoch_x, epoch_y, start, end)
                feed_dict_batch = {x: batch_x, y: batch_y}
                sess.run(optimizer, feed_dict=feed_dict_batch)

                if iteration % display_freq == 0:
                    batch_loss, batch_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict_batch)
                    print("iter {0:3d}:\t Loss={1:.2f},\tTraining Accuracy={2:.01%}".format(
                        iteration, batch_loss, batch_accuracy))
            # Run validation after every epoch
            feed_dict_valid = {x: valid_x[:1000], y: valid_y[:1000]}
            valid_loss, valid_accuracy = sess.run([loss, accuracy], feed_dict=feed_dict_valid)
            print('---------------------------------------------------------')
            print("Epoch: {0}, validation loss: {1:.2f}, validation accuracy: {2:.01%}".format(
                epoch + 1, valid_loss, valid_accuracy))
            print('---------------------------------------------------------')
        path = saver.save(sess, f"{path}/{name}.ckpt")
        # tf.summary.FileWriter(path, tf.get_default_graph())
        print(f"saved at {path}")

        test_x = valid_x[0:10]
        test_y = valid_y[0:10]
        y_ = sess.run(pred, feed_dict={x: test_x})
        print(f"Pred:   {y_}")
        print(f"Actual: {np.argmax(test_y,1)}")


def predict_with_model(x, path=".model"):
    graph = tf.get_default_graph()
    checkpoint = tf.train.latest_checkpoint(path)
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        pred = sess.run("pred:0", feed_dict={"X:0": x})

    return pred


def resize_image(image, size):
    img = PIL.Image.fromarray(np.uint8(image))
    img = img.resize((size, size), PIL.Image.ANTIALIAS)
    # small = np.asarray(img)
    # small = 1 * (small < 8)

    return img


def predict_with_path(path):
    origin = Image.open(path).convert("L")
    inp = np.asarray(origin)
    extracted = Image.fromarray(inp)
    resized = resize_image(extracted, 28)

    img = np.asarray(resized)
    # Invert and Binarize picture
    img = 255.0 * (img < 128)
    img = img.reshape(1, 784)

    return predict_with_model(img)


def retrain():
    train_images, train_labels = load_data_set("/Users/chuck/Kaggle/MNIST/datas", "train", True)
    test_images, test_labels = load_data_set("/Users/chuck/Kaggle/MNIST/datas", "test", True)
    print(train_images.shape, train_labels.shape)
    print(train_images[0].shape, train_labels[0].shape)

    train_images = 255.0 * (train_images > 128)
    test_images = 255.0 * (test_images > 128)
    train_model(train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    # retrain()
    print(predict_with_path(".uploads/image.jpeg"))
