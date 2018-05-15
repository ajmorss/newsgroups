import sys
import os
from cntk import Trainer, Axis
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs,\
        INFINITELY_REPEAT
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk import input_variable, cross_entropy_with_softmax, \
        classification_error, sequence
from cntk.logging import ProgressPrinter
from cntk.layers import Sequential, Embedding, Recurrence, LSTM, Dense
# Define the data dimensions
num_label_classes = 6
import cntk
import numpy as np

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    labelStream = StreamDef(field='label_cat', shape=num_label_classes, is_sparse=False)
    featureStream = StreamDef(field='word', shape=300, is_sparse=False)

    deserializer = CTFDeserializer(path, StreamDefs(labels = labelStream, features = featureStream))

    return MinibatchSource(deserializer,
       randomize = is_training, max_sweeps = 1)


def create_model(x):
    """Create the model for time series prediction"""
    with cntk.layers.default_options(initial_state = 0.1):
        m = Recurrence(LSTM(100))(x)
        m = cntk.sequence.last(m)
        m = cntk.layers.Dropout(0.5)(m)
        m = Dense(1)(m)
        return m


## Define a small model function

x = sequence.input_variable(shape=300, is_sparse=False)
y = input_variable(num_label_classes)

classifier_output = create_model(x)

ce = cross_entropy_with_softmax(classifier_output, y)
pe = classification_error(classifier_output, y)

# Instantiate the trainer object to drive the model training
progress_printer = ProgressPrinter(0)
trainer = Trainer(classifier_output, (ce, pe),
                  cntk.learners.adam(classifier_output.parameters, 0.05, 0.9),
                  progress_printer)

training_loss = []
test_acc = []
step = 0
for i in range(10):
    reader_train = create_reader('../data/train.ctf', True, 300, num_label_classes)
    input_map = {
            x: reader_train.streams.features,
            y:    reader_train.streams.labels
    }

    while True:
        mb = reader_train.next_minibatch(8000, input_map=input_map)
        if len(mb) == 0:
            break
        trainer.train_minibatch(mb)
        training_loss.append([step, trainer.previous_minibatch_loss_average()])
        step = step + 1
    reader_test = create_reader('../data/test.ctf', False, 300, num_label_classes)
    input_map = {
            x: reader_test.streams.features,
            y:    reader_test.streams.labels
    }
    minibatch_acc = []
    while True:
        mb = reader_test.next_minibatch(8000, input_map=input_map)
        if len(mb) == 0:
            break
        minibatch_acc.append(trainer.test_minibatch(mb))
    test_acc.append([i, np.mean(minibatch_acc)])
    print("Done epoch {}".format(str(i)))

import matplotlib.pylab as plt

plt.plot(np.array(training_loss)[:,0], np.array(training_loss)[:.1])
plt.show()

plt.plot(np.array(test_acc)[:,0], np.array(test_acc)[:.1])
plt.show()

