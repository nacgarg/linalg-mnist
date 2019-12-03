import tensorflow as tf
import numpy as np
import pickle


class Model():
    ''' A simple multilayer perceptron with a single hidden layer

    Parameters:
        n_input: Number of input neurons. Default 784
        n_hidden: Size of hidden layer. Default 800
        n_output: Number of output classes. Default 10
        lr: Learning rate. Default 0.00001
    '''

    def __init__(self, n_input=28*28, n_hidden=800, n_output=10, lr=0.001):
        # Initialize dimension parameters
        self._n_input = n_input
        self._n_hidden = n_hidden
        self._n_output = n_output

        # Initialize learning rate
        self._lr = lr

        # Initialize randomized weight and bias matricies
        self._w_hidden = tf.Variable(
            tf.random.normal([self._n_input, self._n_hidden]), name="w_hidden", trainable=True)
        self._w_output = tf.Variable(tf.random.normal(
            [self._n_hidden, self._n_output]), name="w_output", trainable=True)

        self._b_hidden = tf.Variable(
            tf.random.normal([self._n_hidden]), name="b_hidden", trainable=True)
        self._b_output = tf.Variable(
            tf.random.normal([self._n_output]), name="b_output", trainable=True)

        self._trainable_variables = [
            self._w_hidden, self._w_output, self._b_hidden, self._b_output]

    @tf.function
    def _dense(self, x, w, b):
        return tf.matmul(x, w) + b

    @tf.function
    def _forward(self, x):
        ''' Returns forward pass of the network '''

        hidden = self._dense(x, self._w_hidden, self._b_hidden)
        output = self._dense(hidden, self._w_output, self._b_output)
        return tf.math.softmax(output)

    @tf.function
    def _cost(self, forward, y):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=forward, labels=y))

    def train_step(self, x, y):
        ''' Train the network on a single batch '''
        # Compute the loss and gradients of the weights
        with tf.GradientTape() as tape:
            pred = self._forward(x)
            loss = tf.losses.categorical_crossentropy(
                y, pred, from_logits=True)

        gradients = tape.gradient(loss, self._trainable_variables)
        del tape

        # Use the optimizer to apply the gradients
        optimizer = tf.optimizers.Adam(learning_rate=self._lr)
        optimizer.apply_gradients(zip(gradients, self._trainable_variables))
        batch_size = y.shape[0]
        num_correct = 0
        for i in range(batch_size):
            if np.argmax(y[i]) == np.argmax(pred[i]):
                num_correct += 1

        # Return the loss and accuracy so we can display it
        return np.mean(loss[0]), num_correct/batch_size

    def predict_class(self, x):
        # Compute a forward pass, and look up with predicted class
        pred = self._forward(x)
        return np.argmax(pred)

    def save(self, fpath):
        # Save model to disk
        with open(fpath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(fpath):
        # Load model from disk
        with open(fpath, 'rb') as f:
            return pickle.load(f)
