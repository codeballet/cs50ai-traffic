import cv2
from cv2 import IMREAD_COLOR
from cv2 import INTER_AREA
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = list()

    # get a list of all images
    image_paths = list()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            prefix, suffix = file.split('.')
            if suffix == "ppm":
                # only append .ppm files
                image_paths.append(os.path.join(root, file))

    for path in image_paths:
        # add category to labels list
        labels.append(path.split(os.sep)[-2])

        # read each image as a numpy.ndarray
        img = cv2.imread(path, IMREAD_COLOR)

        # resize image
        res = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), INTER_AREA)

        # scale data
        res = res / 255.0

        # add resized image to images list
        images.append(res)

    return (images, labels)


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # create a convolutional neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # convolutional layer, 32 output filters using a 3x3 kernel
    model.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ))

    # max-pooling layer, using 2x2 pool size
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # flatten units
    model.add(tf.keras.layers.Flatten())

    # add hidden layers and dropout
    model.add(tf.keras.layers.Dense(512, activation="relu"))
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.4))

    # output layer
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation="sigmoid"))
    
    model.summary()

    model.compile(
        optimizer="Nadam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    main()
