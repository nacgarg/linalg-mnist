from data import get_mnist_dataset, process_example, process_batch
from model import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import sys
import numpy as np

# Load model from disk or create a new one
if len(sys.argv) > 1:
    m = Model.load(sys.argv[1])
else:
    m = Model(lr=0.001)

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
    m.save("epoch-{}.pkl".format(i))
    print()  # Print empty newline

# Evaluate the model on the test set
results = []
for ex in data['test'].take(1000):
    x, _ = process_example(ex)
    true = ex['label']
    pred = m.predict_class(x)
    results.append(1 if true == pred else 0)

print("Final accuracy on test set: {}".format(sum(results)/len(results)))

# Display a few images with predictions because it's fun
print("\n"*10)
for ex in data['test'].shuffle(100).take(10):
    x, _ = process_example(ex)
    pred = m.predict_class(x)
    print("Predicted: {}".format(pred))
    X = x.reshape((28, 28))
    plt.gray()
    plt.imshow(X)
    plt.waitforbuttonpress()
    plt.close()

plt.close()
# Low pass the loss logs to smoothen the graph
loss_logs = np.convolve(np.array(loss_logs), np.ones((12,))/12, mode='valid')

plt.plot(loss_logs)
plt.show()
