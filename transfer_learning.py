import numpy as np
import cv2
import PIL.Image as Image
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn


class PrepareData:
    """ Prepare the data from https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4 for training """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def get_training_data():

        IMAGE_SHAPE = (224, 224)

        classifier = tf.keras.Sequential([
            hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",
                           input_shape=IMAGE_SHAPE + (3,))
        ])

        # bald_eagle = Image.open('C:/Users/User/Desktop/deep_learning/bald_eagle.jpg').resize(IMAGE_SHAPE).show()
        # bald_eagle = Image.open('C:/Users/User/Desktop/deep_learning/bald_eagle.jpg').resize(IMAGE_SHAPE)
        # bald_eagle = np.array(bald_eagle) / 255.0
        # print(bald_eagle.shape)
        # # bald_eagle = bald_eagle[np.newaxis, :, :]
        # print(bald_eagle.shape)
        # result = classifier.predict(bald_eagle[np.newaxis, :])
        # print(result.shape)
        #
        # predicted_label_index = np.argmax(result)
        # print(predicted_label_index)
        #
        # tf.keras.utils.get_file('ImageNetLabels.txt',
        # 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
        #
        image_labels = []

        with open("ImageNetLabels.txt", "r") as f:
            image_labels = f.read().splitlines()
            print(image_labels[20:25])

        dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
        data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, cache_dir='.', untar=True)
        # print(f'\ndata dir ----- {data_dir}')
        data_dir = pathlib.Path(data_dir)
        # print(f'\npathlib data dir ----- {data_dir}')
        # print(list(data_dir.glob('*/*.jpg'))[:5])
        image_count = len(list(data_dir.glob('*/*.jpg')))
        # print(f'\nNumber of images ----- {image_count}')
        roses = list(data_dir.glob('roses/*'))
        # print(roses[:5])

        roses = list(data_dir.glob('roses/*'))
        tulips = list(data_dir.glob('tulips/*'))
        #
        # for i in range(5):
        #     Image.open(str(roses[i])).show()
        #     Image.open(str(tulips[i])).show()

        # detect turtle img

        # turtle = Image.open('C:/Users/User/Desktop/deep_learning/turtle.jpg').resize(IMAGE_SHAPE).show()
        # turtle = Image.open('C:/Users/User/Desktop/deep_learning/turtle.jpg').resize(IMAGE_SHAPE)
        # turtle = np.array(turtle) / 255.0
        # # print(bald_eagle.shape)
        # bald_eagle = turtle[np.newaxis, :, :]
        # # print(bald_eagle.shape)
        # result = classifier.predict(turtle[np.newaxis, :])
        # # print(result.shape)
        #
        # predicted_label_index = np.argmax(result)
        # print(predicted_label_index)
        #
        # plt.axis('off')
        # plt.imshow(turtle)
        # plt.title(f'\n{image_labels[predicted_label_index]}')
        # plt.show()

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

        # print(flowers_images_dict['roses'][:5])
        # print(str(flowers_images_dict['roses'][0]))

        # img = cv2.imread(str(flowers_images_dict['tulips'][0]))
        # cv2.imshow('Tulips', img)
        # cv2.waitKey()
        # cv2.resize(img, (224, 224))
        # print(img.shape)

        X, y = [], []

        for flower_name, images in flowers_images_dict.items():
            for image in images:
                img = cv2.imread(str(image))
                resized_img = cv2.resize(img, (224, 224))
                X.append(resized_img)
                y.append(flowers_labels_dict[flower_name])

        X = np.array(X)
        y = np.array(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        X_train_scaled = X_train / 255
        X_test_scaled = X_test / 255

        # print(X[0].shape)
        # print(IMAGE_SHAPE+(3,))
        # #
        # # for i in range(4):
        # x0_resized = cv2.resize(X[255], IMAGE_SHAPE)
        # x1_resized = cv2.resize(X[256], IMAGE_SHAPE)
        # x2_resized = cv2.resize(X[257], IMAGE_SHAPE)
        # x3_resized = cv2.resize(X[258], IMAGE_SHAPE)
        #
        # # for i in range(5):
        # #     plt.axis('off')
        # #     plt.imshow(X[i])
        # #     plt.show()
        #
        # predicted = classifier.predict(np.array([x0_resized, x1_resized, x2_resized, x3_resized]))
        # predicted = np.argmax(predicted, axis=1)
        # print(predicted)
        #
        # for i in range(len(predicted)):
        #     plt.axis('off')
        #     plt.imshow(X[255+i])
        #     plt.title(f'\n{image_labels[predicted[i]]}')
        #     plt.show()

        return X_train, y_train, X_test, y_test


class RetrainTheModelWithFlowers:
    """ We take the model from https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4
     and retrain it with flowers pictures """

    def __init__(self, X_train, y_train, X_test, y_test, *args, **kwargs):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.flowers_labels = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']

    def retrain(self):
        feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

        pretrained_model_without_top_layer = hub.KerasLayer(
            feature_extractor_model, input_shape=(224, 224, 3), trainable=False)

        num_of_flowers = 5

        model = tf.keras.Sequential([
            pretrained_model_without_top_layer,
            tf.keras.layers.Dense(num_of_flowers)
        ])

        model.summary()

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['acc'])

        X_train_scaled = self.X_train / 255
        X_test_scaled = self.X_test / 255

        model.fit(X_train_scaled, self.y_train, epochs=5)

        model.evaluate(X_test_scaled, self.y_test)

        y_pred = model.predict(self.X_test)
        y_pred_classes = [np.argmax(element) for element in y_pred]

        print("Classification Report: \n", classification_report(self.y_test, y_pred_classes))

        y_predicted_labels = [np.argmax(i) for i in y_pred]
        cm = tf.math.confusion_matrix(labels=self.y_test, predictions=y_predicted_labels)
        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d', xticklabels=self.flowers_labels, yticklabels=self.flowers_labels)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.show()


def main():

    get_data = PrepareData()
    X_train, y_train, X_test, y_test = get_data.get_training_data()

    retrain = RetrainTheModelWithFlowers(X_train, y_train, X_test, y_test)
    retrain.retrain()


if __name__ == '__main__':
    main()
