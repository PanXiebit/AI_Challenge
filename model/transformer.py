# model for machine translation and question answering

import tensorflow as tf
from model.Modules import *
from model.multi_head_attention import *
import os
from word_vec import train_wordvec


__author__ = "Xie Pan"


class Transformer():
    def __init__(self,d_k, d_v,sentence_len, d_model,
                 vocab_size_cn,
                 vocab_size_en,
                 heads=8,
                 num_layers=6,
                 learning_rate=0.01,
                 dropout_keep_pro=0.5,
                 initializer=tf.random_normal_initializer(0,1)):

        # set hyperparameters
        self.d_k = d_k
        self.d_v = d_v
        self.sentence_len = sentence_len
        self.d_model = d_model   # the dimension of input and output
        self.vocab_size_cn = vocab_size_cn
        self.vocab_size_en = vocab_size_en
        self.heads = heads
        self.num_layers = num_layers   # the number of sub_layers
        self.lr = learning_rate
        self.initializer = initializer
        self.dropout_keep_prob = dropout_keep_pro


        # add placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len],name="input_x") #[None, 30]
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.sentence_len],name="input_y") #[None, 30]
        self.is_training = tf.placeholder(dtype=tf.bool, shape=[], name="is_training")

        # define decoder inputs
        # decoder 的初始输入是随机的？
        self.decoder_inputs = tf.concat([tf.ones_like(self.input_y[:,:1]), self.input_y[:,:-1]],axis=-1) # <start>

        # encoder
        self.enc = self._encoder()

        # decoder
        self.dec = self._decoder()

        # finall linear projection
        self.logits = tf.layers.dense(self.dec, units=self.vocab_size_en) # [batch, sentence_len, vocab_size_en]
        self.prediction = tf.argmax(self.logits, axis=-1, output_type=tf.int32) # [batch, sentence_len]
        self.istarget = tf.to_float(tf.not_equal(self.input_y, 0)) # [1,1,1,1,1,0,0,0,0,0]除去后面padding的部分
        # self.accuracy = tf.cast(tf.equal(self.prediction, tf.argmax(self.input_y)), dtype=tf.float32)
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.prediction, self.input_y)) * self.istarget) / (
            tf.reduce_sum(self.istarget))   # 准确率的计算是看每一个词/总的词的个数
        tf.summary.scalar("accuracy", self.acc)


        # loss and accuracy
        self.y_smoothed = label_smoothing(tf.one_hot(self.input_y, depth=self.vocab_size_en))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
        tf.summary.scalar("loss", self.loss)

        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self.train_step = tf.Variable(0, trainable=False, name='train_step')
        self.train_step = tf.assign(self.train_step, tf.add(self.train_step, tf.constant(1)))
        self.train_op = self.add_train_op()
        tf.summary.scalar('train_step', self.train_step)


    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.9, beta2=0.98, epsilon=1e-9)
        train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return train_op


    def _encoder(self):
        with tf.variable_scope("encoder"):
            # 1. embedding
            with tf.variable_scope("embedding-layer"):
                # 这个embedding还是要修改,没有padding,没有unk
                # self.enc = embedding(inputs=self.input_x,
                #                        vocab_size=self.vocab_size_cn,
                #                        num_units=self.d_model,
                #                        scale=True)   # [batch, sentence_len, d_model]

                # 预训练的embedding
                if not os.path.exists("output/zh_word2vec.npy"):
                    train_wordvec(lan="zh")
                zh_embedding = np.load("output/zh_word2vec.npy")     # [zh_vocab_size, embed_size]
                self.enc = tf.nn.embedding_lookup(zh_embedding, self.input_x)   # [batch_size, sentence_len, embed_size]


            # 2. position encoding
            with tf.variable_scope("position_encoding"):
                encoding = position_encoding_mine(self.enc.get_shape()[1], self.enc.get_shape()[2]) # [sentence_len, embed_size]
                self.enc *= encoding  #[None, 30, 256]

            # 3.dropout
            self.enc = tf.layers.dropout(self.enc,
                                         rate=self.dropout_keep_prob,
                                         training=self.is_training)   # [None, 30, 256]

            # 4. Blocks
            for i in range(self.num_layers):
                with tf.variable_scope("num_layer_{}".format(i)):
                    # multihead attention
                    # encoder: self-attention
                    self.enc = multiheadattention(q=self.enc,
                                                  k=self.enc,
                                                  v=self.enc,
                                                  d_model=self.d_model,
                                                  keys_mask=True,
                                                  heads=self.heads,
                                                  causality=False,
                                                  dropout_keep_prob=self.dropout_keep_prob,
                                                  is_training=self.is_training)  # is_training 在训练和测试的时候不一样,dropout也不一样
                    # Feed Froward
                    self.enc = position_wise_feed_forward(self.enc,
                                                          num_units1= 4*self.d_model,
                                                          num_units2= self.d_model,
                                                          reuse=False)
        return self.enc  #[None, 30, 256]

    def _decoder(self):
        with tf.variable_scope("decoder"):
            # embedding
            # self.dec = embedding(self.decoder_inputs,
            #                     vocab_size=self.vocab_size_en,
            #                     num_units=self.d_model)   # [batch, sentence_len, d_model]
            en_embedding = np.load("output/en_word2vec.npy")                # [en_vocab_size, embed_size]
            self.dec = tf.nn.embedding_lookup(en_embedding, self.input_y)  # [batch, sentence_len, embed_size]

            # position decoding
            encoding = position_encoding_mine(self.dec.get_shape()[1], self.d_model)
            self.dec *= encoding

            # blocks
            for i in range(self.num_layers):
                with tf.variable_scope("num_layers_{}".format(i)):
                    # self-attention
                    with tf.variable_scope("self.attention"):
                        self.dec = multiheadattention(q=self.dec,
                                                      k=self.dec,
                                                      v=self.dec,
                                                      d_model=self.d_model,
                                                      heads=self.heads,
                                                      keys_mask=True,
                                                      causality=True,
                                                      is_training=self.is_training)

                    # encoder-decoder-attention
                    with tf.variable_scope("encoder-decoder-attention"):
                        self.dec = multiheadattention(q=self.dec,
                                                      k=self.enc,
                                                      v=self.enc,
                                                      d_model=self.d_model,
                                                      heads=self.heads,
                                                      keys_mask=True,
                                                      causality=True,
                                                      is_training=self.is_training)

                    self.dec = position_wise_feed_forward(self.dec,
                                                          num_units1= 4*self.d_model,
                                                          num_units2= self.d_model)   # [batch, sentence_len, d_model]

        return self.dec








