import tensorflow as tf


def read_and_decode(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'word_matrix': tf.VarLenFeature(dtype=tf.float32),
            'seq_length': tf.FixedLenFeature(dtype=tf.int64, shape=[]),
            'label_cat': tf.FixedLenFeature(dtype=tf.int64, shape=[6]),
            'label_group': tf.FixedLenFeature(dtype=tf.int64, shape=[20])
            }
        )
    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    word_matrix = tf.reshape(tf.sparse_tensor_to_dense(features['word_matrix']), [-1, 300])
    return word_matrix, features['seq_length'], features['label_cat'], features['label_group']


def get_input_iterator(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_and_decode)
    dataset = dataset.shuffle(5000)
    dataset = dataset.padded_batch(batch_size, ([None, 300], [], [6], [20]))
    iterator = dataset.make_initializable_iterator()
    return iterator, iterator.get_next()
