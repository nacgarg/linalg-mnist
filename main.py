from data import get_mnist_dataset, process_example, process_batch
from model import Model
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import numpy as np

# Load model from disk or create a new one
if len(sys.argv) > 1:
    m = Model.load(sys.argv[1])
else:
    m = Model(lr=0.0002)

# Set up training parameters
data = get_mnist_dataset()
batch_size = 32
epochs = 5
examples = 20000

loss_logs = []

# Main training loop - iterate through data and call train_step
for i in range(epochs):
    num_correct = 0

    print("Epoch: {}".format(i))

    for j, ex in tqdm(enumerate(data['train'].batch(batch_size).take(int(examples/batch_size)))):
        x, y = process_batch(ex)

        loss, accuracy = m.train_step(x, y)

        num_correct += accuracy * batch_size
        loss_logs.append(loss)

        if j % 10 == 1:
            print("Loss: {}, Accuracy {}".format(
                loss, num_correct/(batch_size*(j+1))))
    # Save the model after every epoch
    m.save("epoch-{}.pkl".format(i))

# Evaluate the model on the test set
results = []
for ex in data['test'].take(1000):
    x, _ = process_example(ex)
    true = ex['label']
    pred = m.predict_class(x)
    results.append(1 if true == pred else 0)

print("Final accuracy on test set: {}".format(sum(results)/len(results)))

# Low pass the loss logs to smoothen the graph
loss_logs = np.convolve(np.array(loss_logs), np.ones((8,))/8, mode='valid')

plt.plot(loss_logs)
plt.show()
