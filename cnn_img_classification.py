import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


class Imgs:
    """the images are from the CIFAR10 dataset"""

    def __init__(self, X_train, y_train, X_test, y_test, *args, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.classes = np.array(
            ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        )

    def plot_sample(self, i):
        """ Plot a sample from the CIFAR10 dataset """

        plt.figure(figsize=(5, 5))
        plt.imshow(self.X_train[i])
        plt.xlabel(self.classes[self.y_train[i]])
        plt.show()

    def ann_imgs_classification(self):
        """ Classification of images from the CIFAR10 dataset via an artificial neural network """

        # normalize the training data
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

        # a simple artificial neural network
        ann = models.Sequential([
            layers.Flatten(input_shape=(32, 32, 3)),
            layers.Dense(3000, activation='relu'),
            layers.Dense(1000, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        ann.compile(optimizer='SGD',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        ann.fit(self.X_train, self.y_train, epochs=5)

        self.y_pred = ann.predict(self.X_test)
        self.y_pred_classes = [np.argmax(element) for element in self.y_pred]

        print("Classification Report: \n", classification_report(self.y_test, self.y_pred_classes))

    def cnn_imgs_classification(self):
        """ Classification of images from the CIFAR10 dataset via a simple convolutional neural network """

        cnn = models.Sequential([
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

        cnn.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        cnn.fit(self.X_train, self.y_train, epochs=5)

        cnn.evaluate(self.X_test, self.y_test)

        self.y_pred = cnn.predict(self.X_test)
        print(self.y_pred[:5])
        y_classes = [np.argmax(element) for element in self.y_pred]
        print(y_classes[:5])

        print(f'y test {self.y_test[:5]}')


def main():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_train[0])
    # print(y_train.shape)
    # print(y_train[:5])
    # y_train = y_train.reshape(-1, )
    # print(f'\n reshaped {y_train[:5]}')
    # plt.figure(figsize=(4, 2))
    # plt.imshow(X_train[25])
    # plt.show()

    something = Imgs(X_train, y_train, X_test, y_test)
    # something.plot_sample(0)
    # something.plot_sample(1)
    something.ann_imgs_classification()
    something.cnn_imgs_classification()


if __name__ == '__main__':
    main()
