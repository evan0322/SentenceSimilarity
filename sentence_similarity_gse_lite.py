import tensorflow as tf
import tensorflow_hub as hub
import sentencepiece as spm
import matplotlib.pyplot as plt
import numpy as np
import datetime

import os
import pandas as pd
import re
import seaborn as sns

def process_to_IDs_in_sparse_format(sp, sentences):
  # An utility method that processes sentences with the sentence piece processor
  # 'sp' and returns the results in tf.SparseTensor-similar format:
  # (values, indices, dense_shape)
    ids = [sp.EncodeAsIds(x) for x in sentences]
    max_len = max(len(x) for x in ids)
    dense_shape = (len(ids), max_len)
    values = [item for sublist in ids for item in sublist]
    indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
    return (values, indices, dense_shape)


def plot_similarity(labels, features, rotation):
    print(f"corr finished. {datetime.datetime.now()}")
    corr = np.inner(features, features)
    print(corr)
    sns.set(font_scale=1.2)
    g = sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        cmap="YlOrRd")
    g.set_xticklabels(labels, rotation=rotation)
    g.set_title("Semantic Textual Similarity")


def run_and_plot(session, input_placeholder, messages):
    print(f"run_and_plot started. {datetime.datetime.now()}")

    values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, messages)

    print(f"id processed. {datetime.datetime.now()}")
    message_embeddings = session.run(
        encodings,
        feed_dict={input_placeholder.values: values,
                   input_placeholder.indices: indices,
                   input_placeholder.dense_shape: dense_shape})

    print(f"message_embeddings finished {datetime.datetime.now()}")
    plot_similarity(messages, message_embeddings, 90)


# sen_1 = "Playing 'Do not stop me now'"
# sen_2 = "'Do not stop me now' is playing"

sen_1 = "Playing Game of Thrones Season four episode ten"
sen_2 = "Playing Game of Thrones Season four episode nine"

    # Smartphones
    # "a man is playing a piano",
    # "a man is playing a trump",

    # "Playing 'Do not stop me now'",
    # "Ok, 'Do not stop me now' is playing",
    # "Playing 'Do not stop me now'",
    # "'Do not stop me now' is playing",
    #     "a man is cutting up a cucumber",
    # "a man is slicing a cucumber",
    # "the dog bites the man",
    # "the man bites the dog",
    # "Your cellphone looks great.",
    #
    # # Weather
    # "Will it snow tomorrow?",
    # "Recently a lot of hurricanes have hit the US",
    # "Global warming is real",
    #
    # # Food and health
    # "An apple a day, keeps the doctors away",
    # "Eating strawberries is healthy",
    # "Is paleo better than keto?",
    #
    # # Asking about age
    # "How old are you?",
    # "what is your age?",

messages = [
    sen_1,
    sen_2
]


module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")

print(f"Execution started. {datetime.datetime.now()}")

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
encodings = module(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))


with tf.Session() as sess:
    spm_path = sess.run(module(signature="spm_path"))

sp = spm.SentencePieceProcessor()
sp.Load(spm_path)
print("SentencePiece model loaded at {}.".format(spm_path))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.tables_initializer())
    run_and_plot(session, input_placeholder, messages)
