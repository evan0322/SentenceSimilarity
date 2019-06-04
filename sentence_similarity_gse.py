import tensorflow as tf
import tensorflow_hub as hub
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns


module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

print(f"Execution started. {datetime.datetime.now()}")

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  print(corr)
  print(f"corr finished. {datetime.datetime.now()}")
  # sns.set(font_scale=1.2)
  # g = sns.heatmap(
  #     corr,
  #     xticklabels=labels,
  #     yticklabels=labels,
  #     vmin=0,
  #     vmax=1,
  #     cmap="YlOrRd")
  # g.set_xticklabels(labels, rotation=rotation)
  # g.set_title("Semantic Textual Similarity")


def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
  message_embeddings_ = session_.run(encoding_tensor, feed_dict={input_tensor_: messages_})
  print(f"Embedding finished. {datetime.datetime.now()}")
  plot_similarity(messages_, message_embeddings_, 90)


messages = [
    "a man is not playing a piano",
    "a man is playing a piano",
    # "a man is cutting up a cucumber",
    # "a man is slicing a cucumber",
    # "the dog bites the man",
    # "the man bites the dog",
    # Smartphones
    # "I like my phone",
    # "My phone is not good.",
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
]

similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
similarity_message_encodings = embed(similarity_input_placeholder)

print(f"Model initialized. {datetime.datetime.now()}")

with tf.Session() as session:
  print(f"Session started. {datetime.datetime.now()}")
  session.run(tf.global_variables_initializer())
  session.run(tf.tables_initializer())
  print(f"Session initialized. {datetime.datetime.now()}")
  saver = tf.train.Saver()
  saver.save(session, 'my_test_model')

  # with tf.Session() as sess:
  #     saver.restore(sess, "my_test_model")
  #     run_and_plot(sess, similarity_input_placeholder, messages,similarity_message_encodings)
  run_and_plot(session, similarity_input_placeholder, messages, similarity_message_encodings)
# def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
