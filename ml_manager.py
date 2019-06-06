import tensorflow as tf
import tensorflow_hub as hub
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import sentencepiece as spm
import datetime

class MLManager:
    def __init__(self, model_type):
        self.model_type = model_type
        self.module = self.create_tf_module(model_type)
        self.session, self.sp = self.create_tf_session(model_type, self.module)

    def create_tf_session(self, model_type, module):
        return self.create_tf_session_lite(module)

    def create_tf_module(self, model_type):
        module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        return module

    def create_tf_session_lite(self, module):
        session = tf.Session()
        spm_path = session.run(module(signature="spm_path"))
        sp = spm.SentencePieceProcessor()
        sp.Load(spm_path)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        return session, sp

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

    def get_STS_score(self, sentence1, sentence2):
        input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])
        encodings = self.module(inputs=dict(values=input_placeholder.values,indices=input_placeholder.indices,dense_shape=input_placeholder.dense_shape))
        values, indices, dense_shape = self.process_to_IDs_in_sparse_format(self.sp, [sentence1, sentence2])
        message_embeddings = self.session.run(
        encodings,feed_dict={input_placeholder.values: values,
                   input_placeholder.indices: indices,
                   input_placeholder.dense_shape: dense_shape})

        corr = np.inner(message_embeddings, message_embeddings)
        return corr
