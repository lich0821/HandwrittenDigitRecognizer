#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
import tensorflow as tf

import os
import numpy as np
from PIL import Image
from recognizer import load_data_set, shuffle_samples, get_next_batch, resize_image
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class HDRCNN():
    def __init__(self, path=".model/cnn"):
        self.path = path
        global graph                    # Fix Flask multithread problem
        graph = tf.get_default_graph()  # Fix Flask multithread problem
        if os.path.isfile(path):
            self.model = load_model(path)
        else:
            x_train, y_train = load_data_set("/Users/chuck/Kaggle/MNIST/datas", "train", True)
            x_test, y_test = load_data_set("/Users/chuck/Kaggle/MNIST/datas", "test", True)

            x_train = x_train.reshape(-1, 28, 28, 1)
            x_test = x_test.reshape(-1, 28, 28, 1)

            x_train /= 255.0
            x_test /= 255.0

            self.model = self.train_model(x_train, y_train, x_test, y_test)

    def train_model(self, x_train, y_train, x_test, y_test, epochs=12, batch_size=128):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=x_train.shape[1:]))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        # print(model.summary())
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print("Test loss:     ", score[0])
        print("Test accuracy: ", score[1])

        idx = int(score[1] * 10000)
        print(idx)
        pred = model.predict_classes(x_test[idx:idx + 20], 20, 1)
        print("Predict: ", pred)
        print("Actual:  ", np.argmax(y_test[idx:idx + 20], axis=1))

        model.save(self.path)
        return model

    def predict_with_model(self, x, path=".model/cnn"):
        x = x.reshape(-1, 28, 28, 1)
        with graph.as_default():  # Fix Flask multithread problem
            pred = self.model.predict_classes(x)

        return pred

    def predict_with_path(self, path):
        origin = Image.open(path).convert("L")
        inp = np.asarray(origin)
        extracted = Image.fromarray(inp)
        resized = resize_image(extracted, 28)

        img = np.asarray(resized)
        img = (255 - img) / 255.0

        return self.predict_with_model(img)


cnn = HDRCNN()

if __name__ == "__main__":
    pred = cnn.predict_with_path(".uploads/image.jpeg")
    print(pred)
