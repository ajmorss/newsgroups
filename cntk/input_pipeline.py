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
    featureStream = StreamDef(field='word_matrix', shape=300, is_sparse=False)
    seqlenStream = StreamDef(field='seq_length', shape=1, is_sparse=False)

    deserializer = CTFDeserializer(path, StreamDefs(labels = labelStream, features = featureStream, seqlen = seqlenStream))

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
            Recurrence(LSTM(50), go_backwards=False),
            Recurrence(LSTM(50), go_backwards=False),
            Dense(num_label_classes, name='output')
        ])
        return cntk.sequence.last(m)



## Define a small model function

x = sequence.input_variable(shape=300, is_sparse=True)
y = input_variable(num_label_classes)

classifier_output = create_model_mn_factory()
classifier_output = classifier_output(x)

ce = cross_entropy_with_softmax(classifier_output, y)
pe = classification_error(classifier_output, y)

reader = create_reader('../data/train.ctf', True,300, num_label_classes)

input_map = {
        x: reader.streams.features,
        y:    reader.streams.labels
}

lr_per_sample = learning_rate_schedule(0.0005, UnitType.sample)
# Instantiate the trainer object to drive the model training
progress_printer = ProgressPrinter(0)
trainer = Trainer(classifier_output, (ce, pe),
                  sgd(classifier_output.parameters, lr=lr_per_sample),
                  progress_printer)


for i in range(255):
    mb = reader.next_minibatch(64, input_map=input_map)
    trainer.train_minibatch(mb)
