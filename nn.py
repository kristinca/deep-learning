import numpy as np


def sigmoid_numpy(x):
    return 1 / (1 + np.exp(-x))


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i, epsilon) for i in y_predicted]
    print(y_predicted_new)
    y_predicted_new = [min(i, 1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    print(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new) + (1-y_true)*(np.log(1-y_predicted_new)))


class myNN:
    """A simple neural network"""
    def __init__(self, *args, **kwargs):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self, x, y, epochs, loss_threshold):
        self.gradient_descent(x['age'], x['affordability'], y, epochs, loss_threshold)

    def gradient_descent(self, age, affordability, y_true, epochs, loss_threshold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)

            w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d

            if i % 50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss <= loss_threshold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias
