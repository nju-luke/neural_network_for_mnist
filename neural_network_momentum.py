# -*- coding: utf-8 -*-
# @Time    : 30/12/2017
# @Author  : Luke

from utils import *


class Config():
    lr = 0.1
    lr_decay_rate = 0.5
    gama = 1e-5         #normalization
    beta = 0.9          #momentum

    input_size = 784
    labels_size = 10
    layers_size = [1024, 128]

    activation = 'relu'
    epoch = 32
    batch_size = 128

    is_test = False


activation = eval(Config.activation)
g_activation = eval("g_" + Config.activation)


class Dense():
    def __init__(self, input_size, hidden_size, config):
        self.W, self.b = parameters_init(input_size, hidden_size)
        self.config = config
        self.v_W = np.zeros(np.shape(self.W))
        self.v_b = np.zeros(np.shape(self.b))

    def output(self, X):
        self.input = X
        self.batch_size = len(X)
        self.z = np.dot(X, self.W) + self.b
        self.a = activation(self.z)
        self.g_activation = g_activation
        return self.a

    def upgrade(self, delta):
        delta = delta * self.g_activation(self.z)

        dW = np.dot(self.input.T, delta) / self.batch_size
        self.v_W = self.config.beta * self.v_W + (1-self.config.beta)*dW
        self.W -= self.config.lr * dW + self.config.gama * self.W

        db = np.mean(delta, axis=0, keepdims=True)
        self.v_b = self.config.beta * self.v_b + (1-self.config.beta)*db
        self.b -= self.config.lr * self.v_b

        return np.dot(delta, self.W.T)


class Softmax(Dense):
    def __init__(self, input_size, labels_size, config):
        super().__init__(input_size, labels_size, config)

    def output(self, X):
        self.input = X
        self.batch_size = len(X)
        self.z = np.dot(X, self.W) + self.b
        self.a = softmax(self.z)
        self.g_activation = g_softmax
        return self.a


class Model():
    def __init__(self, config):
        self.config = config
        self.build_model()

    def get_batch(self, X, y):
        size = len(X)
        assert size == len(y)
        indices = np.arange(size)
        np.random.shuffle(indices)

        i = 0
        while i * self.config.batch_size < size:
            ids = indices[
                  i * self.config.batch_size:(i + 1) * self.config.batch_size]
            yield X[ids], y[ids]
            i += 1

    def build_model(self):
        self.layers = []
        for i in range(len(self.config.layers_size)):
            if i == 0:
                input_size = self.config.input_size
                output_size = self.config.layers_size[i]
            else:
                input_size = self.config.layers_size[i - 1]
                output_size = self.config.layers_size[i]
            layer = Dense(input_size, output_size, self.config)
            self.layers += [layer]
        self.layer_output = Softmax(self.config.layers_size[-1],
                                    self.config.labels_size, self.config)

    def loss(self, y, y_pred):
        loss = cross_entropy(y, y_pred) / len(y)
        loss += np.sum(self.layer_output.W ** 2) * self.config.gama
        for layer in self.layers:
            loss += np.sum(layer.W ** 2) * self.config.gama
        return loss

    def forward(self, X):
        for layer in self.layers:
            X = layer.output(X)
        logits = self.layer_output.output(X)
        return logits

    def backward(self, y, y_pred):
        delta = y_pred - y
        delta = self.layer_output.upgrade(delta)
        for i in range(len(self.layers))[::-1]:
            delta = self.layers[i].upgrade(delta)

    def get_accuracy(self, X, y):
        y_pred = self.predict(X)
        acc = np.sum(y_pred == y) / len(y)
        return acc

    def predict(self, X):
        if len(np.shape(X)) == 1:
            X = np.reshape(X, (1, -1))
        logits = self.forward(X)
        return np.argmax(logits, axis=1)

    def fit(self):
        train_data, train_label, test_data, test_label = load_data(
            self.config.is_test)
        train_label_gt = ground_truth(train_label)

        loss_history = []
        for j in range(self.config.epoch):
            loss_ = []
            batches = self.get_batch(train_data, train_label_gt)
            for i, (X, y) in enumerate(batches):
                logits = self.forward(X)
                loss = self.loss(y, logits)
                loss_.append(loss)
                self.backward(y, logits)
                if i % 100 == 0:
                    print("Stage {}, {}".format(i * self.config.batch_size,
                                                np.mean(loss_)))

            print("\nEpoch: %s, loss = %s" % (j, np.mean(loss_)))

            acc = self.get_accuracy(train_data, train_label)
            print("Acc of train data : %.3f" % acc)
            acc = self.get_accuracy(test_data, test_label)
            print("Acc of test data : %.3f\n" % acc)

            loss_history.append(np.mean(loss_))

            if j > 3 and loss_history[-1] > loss_history[-2]:
                self.config.lr *= self.config.lr_decay_rate
                print("The learning rate has been decayed to: "
                      "%s" % self.config.lr)
            if self.config.lr < 1e-6:
                break


config = Config()
model = Model(config)
model.fit()
