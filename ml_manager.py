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
        print("model started")
        # self.model_type = model_type
        # # self.module = self.create_tf_module(model_type)
        # # self.session, self.sp = self.create_tf_session(model_type, self.module)
        #
        # module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
        # self.session = tf.Session()
        # spm_path = self.session.run(module(signature="spm_path"))
        # self.sp = spm.SentencePieceProcessor()
        # self.sp.Load(spm_path)
        # self.session.run(tf.global_variables_initializer())
        # self.session.run(tf.tables_initializer())

        #large
        # self.session = tf.Session()
        # self.session.run(tf.global_variables_initializer())
        # self.session.run(tf.tables_initializer())

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

    def process_to_IDs_in_sparse_format(self, sp, sentences):
        # An utility method that processes sentences with the sentence piece processor
        # 'sp' and returns the results in tf.SparseTensor-similar format:
         # (values, indices, dense_shape)
        ids = [sp.EncodeAsIds(x) for x in sentences]
        max_len = max(len(x) for x in ids)
        dense_shape = (len(ids), max_len)
        values = [item for sublist in ids for item in sublist]
        indices = [[row, col] for row in range(len(ids)) for col in range(len(ids[row]))]
        return (values, indices, dense_shape)

    def get_STS_score_lite(self, sentence1, sentence2):
        module = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2")
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
            messages = [sentence1, sentence2]
            values, indices, dense_shape = self.process_to_IDs_in_sparse_format(sp, messages)
            message_embeddings = session.run(
                encodings,
                feed_dict={input_placeholder.values: values,
                        input_placeholder.indices: indices,
                        input_placeholder.dense_shape: dense_shape})
            corr = np.inner(message_embeddings, message_embeddings)
            return corr.item((0,1))

        # encodings = module(inputs=dict(values=input_placeholder.values,indices=input_placeholder.indices,dense_shape=input_placeholder.dense_shape))
        # message = [sentence1, sentence2]
        # values, indices, dense_shape = self.process_to_IDs_in_sparse_format(self.sp, message)
        # message_embeddings = self.session.run(
        # encodings,feed_dict={input_placeholder.values: values,
        #            input_placeholder.indices: indices,
        #            input_placeholder.dense_shape: dense_shape})
        #
        # corr = np.inner(message_embeddings, message_embeddings)
        # return corr


    def get_STS_score_large(self, sentence1, sentence2):
        with tf.Session() as session:

            session.run(tf.global_variables_initializer())
            session.run(tf.tables_initializer())


            print('inside get sts')
            module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]
        # Import the Universal Sentence Encoder's TF Hub module
            embed = hub.Module(module_url)
            print('inside get sts 1')

            messages = [sentence1, sentence2]
            similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
            similarity_message_encodings = embed(similarity_input_placeholder)
            message_embeddings_ = session.run(similarity_message_encodings, feed_dict={similarity_input_placeholder: messages})
            corr = np.inner(message_embeddings_, message_embeddings_)

            print('inside get sts 2')

            return  corr



# manager = MLManager("large")
# manager.get_STS_score_lite("wei has an apple", "wei has an orange")
