from data import get_mnist_dataset, process_example, process_batch
from model import Model
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import sys
import numpy as np

def tune(n_hidden_in, dropout_in):
    print("Tuning with Hidden Size = {}, Dropout = {}".format(n_hidden_in, dropout_in))
    m = Model(n_hidden = n_hidden_in, dropout = dropout_in)
    # Set up training parameters
    data = get_mnist_dataset()
    batch_size = 32
    epochs = 15
    examples = 60000

    train_data = data['train'].shuffle(examples, reshuffle_each_iteration=False)

    loss_logs = []

    # Main training loop - iterate through data and call train_step
    for i in range(epochs):
        num_correct = 0

        for j, ex in enumerate(train_data.batch(batch_size).take(int(examples/batch_size))):
            x, y = process_batch(ex)

            loss, accuracy = m.train_step(x, y)

            num_correct += accuracy * batch_size
            loss_logs.append(loss)

            print("Epoch {}, {}/{}".format(i, (j+1)*batch_size, examples) + " "*10 + "Loss: {}, Accuracy {}".format(
                loss, num_correct/(batch_size*(j+1))), end='\r', flush=True)
        # Save the model after every epoch
        # m.save("epoch-{}.pkl".format(i))
        print()  # Print empty newline

    # Evaluate the model on the test set
    results = []
    for ex in data['test'].take(1000):
        x, _ = process_example(ex)
        true = ex['label']
        pred = m.predict_class(x)
        results.append(1 if true == pred else 0)

    # Low pass the loss logs to smoothen the graph
    loss_logs = np.convolve(np.array(loss_logs), np.ones((12,))/12, mode='valid')

    plt.plot(loss_logs)
    plt.savefig("figure-hidden{}-dropout{}-accuracy{}.png".format(n_hidden_in, dropout_in, sum(results)/len(results)), dpi = 400)
    plt.clf()


if __name__ == "__main__":
    for i in range(200, 400, 100):
        for j in range(20, 45, 5):
            tune(i, j/100)

