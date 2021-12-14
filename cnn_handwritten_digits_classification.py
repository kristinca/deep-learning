import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sn


class Numbers:
    """ The handwritten digits are from the MNIST dataset """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def cnn_number_classification(self):
        """ Classification of handwritten digits from the MNIST dataset via a simple convolutional neural network """

        cnn = models.Sequential([
                layers.Conv2D(filters=20, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.MaxPooling2D((2, 2)),

                layers.Flatten(),
                layers.Dense(10, activation='sigmoid')
        ])

        cnn.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        cnn.fit(self.X_train, self.y_train, epochs=5)

        cnn.evaluate(self.X_test, self.y_test)

        self.y_pred = cnn.predict(self.X_test)
        self.y_pred_classes = [np.argmax(element) for element in self.y_pred]

        print("Classification Report: \n", classification_report(self.y_test, self.y_pred_classes))

        y_predicted_labels = [np.argmax(i) for i in self.y_pred]
        cm = tf.math.confusion_matrix(labels=self.y_test, predictions=y_predicted_labels)

        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()


def main():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Set input shape
    sample_shape = X_train[0].shape
    img_width, img_height = sample_shape[0], sample_shape[1]
    input_shape = (img_width, img_height, 1)

    # Reshape data
    X_train = X_train.reshape(len(X_train), input_shape[0], input_shape[1], input_shape[2])
    X_test = X_test.reshape(len(X_test), input_shape[0], input_shape[1], input_shape[2])

    some = Numbers(X_train, y_train, X_test, y_test)
    some.cnn_number_classification()


if __name__ == '__main__':
    main()
