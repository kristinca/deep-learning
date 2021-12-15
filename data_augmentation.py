import cv2
import os
from PIL import Image
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib


class Prepare_training_data:
    """ Prepare the data from
     https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz for training """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_training_data():

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
        # cache_dir indicates where to download data. I specified . which means current directory
        # untar true will unzip it
        # print(data_dir)
        data_dir = pathlib.Path(data_dir)
        # print(data_dir)
        # print(list(data_dir.glob('*/*.jpg'))[:5])

        image_count = len(list(data_dir.glob('*/*.jpg')))
        # print(f'No. of images = {image_count}')

        roses = list(data_dir.glob('roses/*'))
        tulips = list(data_dir.glob('tulips/*'))

        # for i in range(5):
        #     PIL.Image.open(str(roses[i])).show()
        #     PIL.Image.open(str(tulips[i])).show()

        flowers_images_dict = {
            'roses': list(data_dir.glob('roses/*')),
            'daisy': list(data_dir.glob('daisy/*')),
            'dandelion': list(data_dir.glob('dandelion/*')),
            'sunflowers': list(data_dir.glob('sunflowers/*')),
            'tulips': list(data_dir.glob('tulips/*')),
        }

        # assign a class number to each of these flowers

        flowers_labels_dict = {
            'roses': 0,
            'daisy': 1,
            'dandelion': 2,
            'sunflowers': 3,
            'tulips': 4,
        }
        #
        # print(flowers_images_dict['roses'][:5])
        # str(flowers_images_dict['roses'][0])
        # print(flowers_images_dict['roses'][100])

        img = cv2.imread(str(flowers_images_dict['roses'][100]))
        # cv2.imshow('A rose', img)

        # cv2.waitKey(0)

        # print(f'\noriginal image coverted to 3 D numpy array ----- {img.shape}')
        # print(f'\nresized numpy array of the converted image (resized img) ---- {cv2.resize(img, (180, 180)).shape}')

        X, y = [], []

        for flower_name, images in flowers_images_dict.items():
            # print(f'\n{flower_name}')
            # print(len(images))
            for image in images:
                img = cv2.imread(str(image))
                resized_img = cv2.resize(img, (180, 180))
                X.append(resized_img)
                # the number for each flower: from flowers labels dict :
                y.append(flowers_labels_dict[flower_name])
            # print(f'{y[:5]}, \n {X}')
        X = np.array(X)
        y = np.array(y)
        # print(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        return X_train, y_train, X_test, y_test


class Flowers:
    """ The flowers pics are from
    https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz """

    def __init__(self, X_train, y_train, X_test, y_test, *args, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.flowers_labels = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

    def cnn_flowers_classification(self):
        """ Convolutional neural network classification of flowers from
        https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz """

        # there are 5 types of flowers
        num_classes = 5

        self.X_train_scaled = self.X_train / 255
        self.X_test_scaled = self.X_test / 255

        cnn = Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            # layers.Conv2D(64, 3, padding='same', activation='relu'),
            # layers.MaxPooling2D(),
            layers.Flatten(),
            # layers.Dense(128, activation='sigmoid'),
            layers.Dense(num_classes, activation='sigmoid')
        ])

        cnn.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])

        cnn.fit(self.X_train_scaled, self.y_train, epochs=5)
        cnn.evaluate(self.X_test, self.y_test)

        y_pred = cnn.predict(self.X_test)
        y_pred_classes = [np.argmax(element) for element in y_pred]

        print("Classification Report: \n", classification_report(self.y_test, y_pred_classes))

        y_predicted_labels = [np.argmax(i) for i in y_pred]
        cm = tf.math.confusion_matrix(labels=self.y_test, predictions=y_predicted_labels)
        # cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d', xticklabels=self.flowers_labels, yticklabels=self.flowers_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()


def main():
    get_data = Prepare_training_data()

    X_train, y_train, X_test, y_test = get_data.get_training_data()
    print(f'No. of images for training the model ----- {len(X_train)} \nNo. of images for testing the model '
          f'----- {len(X_test)}')

    flowers = Flowers(X_train, y_train, X_test, y_test)

    flowers.cnn_flowers_classification()


if __name__ == '__main__':
    main()
