from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import gensim
import glob
import codecs

import tensorflow as tf


def get_data():
    import urllib
    data_getter = urllib.URLopener()
    data_getter.retrieve("http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz", "20news-bydate.tar.gz")

    import os
    os.mkdir("data")

    import tarfile
    tar = tarfile.open("20news-bydate.tar.gz", "r:gz")
    tar.extractall(path="data")
    tar.close()


def save_example(doc, labels, writer):
    word_matrix = np.reshape(doc, -1)
    word_matrix = tf.train.Feature(
        float_list=tf.train.FloatList(value=word_matrix)
    )

    seq_length = np.array([doc.shape[0]]).astype(int)
    seq_length = tf.train.Feature(
        int64_list=tf.train.Int64List(value=seq_length)
    )

    label_cat = np.zeros(6).astype(int)
    label_cat[labels[0]] = 1
    label_cat = tf.train.Feature(
        int64_list=tf.train.Int64List(value=label_cat)
    )

    label_group = np.zeros(20).astype(int)
    label_group[labels[1]] = 1
    label_group = tf.train.Feature(
        int64_list=tf.train.Int64List(value=label_group)
    )

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'word_matrix': word_matrix,
                'seq_length': seq_length,
                'label_cat': label_cat,
                'label_group': label_group
                }
            )
        )
    writer.write(example.SerializeToString())


def clean_doc(doc):
    doc = doc.lower()
    doc = gensim.parsing.preprocessing.remove_stopwords(doc)
    doc = gensim.utils.lemmatize(doc)
    doc = [x.split('/')[0] for x in doc]
    return doc


def vectorize_doc(doc, model):
    doc_out = np.zeros((len(doc), 300))
    for i in xrange(len(doc)):
        try:
            doc_out[i, :] = model[doc[i]]
        except:
            doc_out[i, :] = model['UNK']
    return doc_out


def get_labels(fname):
    cats = {'comp.graphics': 0,
            'comp.os.ms-windows.misc': 0,
            'comp.sys.ibm.pc.hardware': 0,
            'comp.sys.mac.hardware': 0,
            'comp.windows.x': 0,
            'rec.autos': 1,
            'rec.motorcycles': 1,
            'rec.sport.baseball': 1,
            'rec.sport.hockey': 1,
            'sci.crypt': 2,
            'sci.electronics': 2,
            'sci.med': 2,
            'sci.space': 2,
            'talk.politics.misc': 3,
            'talk.politics.guns': 3,
            'talk.politics.mideast': 3,
            'talk.religion.misc': 4,
            'alt.atheism': 4,
            'soc.religion.christian': 4,
            'misc.forsale': 5}
    groups = {'comp.graphics': 0,
              'comp.os.ms-windows.misc': 1,
              'comp.sys.ibm.pc.hardware': 2,
              'comp.sys.mac.hardware': 3,
              'comp.windows.x': 4,
              'rec.autos': 5,
              'rec.motorcycles': 6,
              'rec.sport.baseball': 7,
              'rec.sport.hockey': 8,
              'sci.crypt': 9,
              'sci.electronics': 10,
              'sci.med': 11,
              'sci.space': 12,
              'talk.politics.misc': 13,
              'talk.politics.guns': 14,
              'talk.politics.mideast': 15,
              'talk.religion.misc': 16,
              'alt.atheism': 17,
              'soc.religion.christian': 18,
              'misc.forsale': 19}
    group = fname.split('/')[2]
    return cats[group], groups[group]


def preprocess(model):
    writer_train = tf.python_io.TFRecordWriter('data/train.tfrecord')
    writer_test = tf.python_io.TFRecordWriter('data/test.tfrecord')
    #writer_test = tf.python_io.TFRecordWriter(tf_record_fname_test)
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    fnames = glob.glob('data/*/*/*')
    for x in fnames:
        print(x)
        with codecs.open(x, "r",encoding='utf-8', errors='ignore') as fdata:
            doc = fdata.read()
        doc = clean_doc(doc)
        doc = vectorize_doc(doc, model)
        labels = get_labels(x)
        if 'test' in x:
            save_example(doc, labels, writer_test)
        else:
            save_example(doc, labels, writer_train)


if __name__ == '__main__':
    try:
        model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    except:
        print("Please download GoogleNews-vectors-negative300 from: \n"
              "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit \n"
              "unzip it and place it in this directory.")
        raise
    if len(glob.glob('data/*/*/*')) == 0:
        get_data()

    preprocess(model)
