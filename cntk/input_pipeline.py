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

# Read a CTF formatted text (as mentioned above) using the CTF deserializer from a file
def create_reader(path, is_training, input_dim, num_label_classes):

    labelStream = StreamDef(field='label_cat', shape=num_label_classes, is_sparse=False)
    featureStream = StreamDef(field='word', shape=300, is_sparse=False)

    deserializer = CTFDeserializer(path, StreamDefs(labels = labelStream, features = featureStream))

    return MinibatchSource(deserializer,
       randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)


def LSTM_sequence_classifier_net(input, num_output_classes, embedding_dim,
                                LSTM_dim, cell_dim):
    lstm_classifier = Sequential([Embedding(embedding_dim),
                                  Recurrence(LSTM(LSTM_dim, cell_dim)),
                                  sequence.last,
                                  Dense(num_output_classes)])
    return lstm_classifier(input)

def create_model_mn_factory():
    with cntk.layers.default_options(initial_state=0.1):
        m = Sequential([
            #Recurrence(LSTM(50), go_backwards=False),
            Recurrence(LSTM(100), go_backwards=False),
            cntk.layers.Dropout(0.5, seed=1),
            Dense(num_label_classes, name='output')
        ])
        return cntk.sequence.last(m)



## Define a small model function

x = sequence.input_variable(shape=300, is_sparse=False)
y = input_variable(num_label_classes)

classifier_output = create_model_mn_factory()
classifier_output = classifier_output(x)

ce = cross_entropy_with_softmax(classifier_output, y)
pe = classification_error(classifier_output, y)

reader = create_reader('../data/train.ctf', True, 300, num_label_classes)

input_map = {
        x: reader.streams.features,
        y:    reader.streams.labels
}

lr_per_sample = cntk.learners.learning_parameter_schedule([0.01, 0.01, 0.01, 0.05, 0.05, .01, 0.01, 0.005, 0.005, 0.001], minibatch_size=8000, epoch_size=150)
# Instantiate the trainer object to drive the model training
progress_printer = ProgressPrinter(0)
trainer = Trainer(classifier_output, (ce, pe),
                  cntk.learners.adam(classifier_output.parameters, 0.01, 0.9),
                  progress_printer)

for i in range(15000):
    mb = reader.next_minibatch(8000, input_map=input_map)
    trainer.train_minibatch(mb)


