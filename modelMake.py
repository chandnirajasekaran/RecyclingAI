import cv2
import numpy as np
import os
import sys
import tensorflow as tf


from sklearn.model_selection import train_test_split


EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 2
TEST_SIZE = 0.3


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
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file

    model.save('modelLoadCategorical.h5')


def load_data(data_dir):
    """
    Load image data from directory `data_dir`
    """
    images = []
    labels = []

    for x in range(2):
        directory = os.path.join("/Users", "akashgujjar", "PycharmProjects", "recyclingAI", "recycleData", str(x))
        count = 0

        for file in os.listdir(directory):
            im = cv2.imread(os.path.join(directory, file))
            if file[-1] == 'e':
                continue
            resized_image = cv2.resize(im, (IMG_HEIGHT, IMG_WIDTH))
            count += 1
            images.append(resized_image)
            labels.append(str(x))

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model
    """
    model = tf.keras.models.Sequential([
        # input layer
        tf.keras.layers.Conv2D(60, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(axis=2),

        tf.keras.layers.Conv2D(80, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3)),
        tf.keras.layers.BatchNormalization(axis=2),

        tf.keras.layers.Conv2D(90, (2, 2), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # output layer
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    print(model)
    return model


if __name__ == "__main__":
    main()
