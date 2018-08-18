import tensorflow as tf
import os
import numpy as np

class seq2seq(object):
    def __init__(self, sequence_len, vocab_size_x, vocab_size_y, embed_size, num_units, rnn_type,
                 batch_size, pretrained_wordvec, learning_rate, max_gradient_norm=1):
        self.sequence_len = sequence_len
        self.vocab_size_x = vocab_size_x
        self.vocab_size_y = vocab_size_y
        self.embed_size = embed_size
        self.num_units = num_units
        self.rnn_type = rnn_type
        self.batch_size = batch_size
        self.pretrained_wordvec = pretrained_wordvec
        self.learning_rate =learning_rate
        self.max_gradient_norm = max_gradient_norm

        # add placeholder
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_len], name="source-sentnece")
        self.input_y = tf.placeholder(dtype=tf.int32, shape=[None, self.sequence_len], name="target-sequence")
        self.is_training = tf.placeholder(dtype=tf.bool, name="is_training")
        
        # encoder
        self.encoder_state = self._encoder()

        # decoder
        self.decoder_output = self._decoder_train()  # the length is varied in inference/test

        # accuracy
        self.logits = tf.layers.dense(self.decoder_output, units=self.vocab_size_y, name="logits")   # [None, sequence_len, vocab_size_y]
        self.prediction = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name="prediction")   # [None, sequence_len]
        self.target_weights = tf.cast(tf.not_equal(self.input_y, 0), tf.float32, name="target-weights")   # [None, sequence_len] 不是 0 的位置设为1, 除去 padding 的影响
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.prediction, self.input_y), tf.float32)*self.target_weights)\
                        / tf.reduce_sum(self.target_weights)
        self.acc = tf.cast(self.accuracy/self.batch_size , tf.float32, name= "accuracy")
        tf.summary.scalar("accuracy", self.acc)

        # loss
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=tf.one_hot(self.input_y, self.vocab_size_y), dim=-1) # [None, sequence_len]
        self.loss = tf.reduce_sum(self.cross_entropy * self.target_weights) / self.batch_size
        tf.summary.scalar("train_loss", self.loss)

        # train using adam
        self.train_op = self._train_op()
        
    def _encoder(self):
        with tf.variable_scope("encoder"):
            with tf.variable_scope("embedding-layer"):
                if not self.pretrained_wordvec:
                    embedding_encoder = tf.get_variable(name="decoder-embedding",
                                                        shape=[self.vocab_size_x, self.embed_size], dtype=tf.float32)
                else:
                    embedding_encoder = np.load("glove.npy")
                tf.summary.histogram(name="embedding-encoder",values=embedding_encoder)
                embeded_x = tf.nn.embedding_lookup(embedding_encoder, self.input_x)  # [None, sequence_len, embed_size]

            with tf.variable_scope("backward-rnn"):
                # backward rnn
                embeded_x_back = tf.reverse_v2(embeded_x, axis=[1])
                self.encoder_rnn_cell = self._get_cell(self.num_units, self.rnn_type)
                init_state = self.encoder_rnn_cell.zero_state(self.batch_size, dtype=tf.float32)
                encoder_output, encoder_state = tf.nn.dynamic_rnn(cell=self.encoder_rnn_cell,
                                                                  inputs=embeded_x_back,
                                                                  initial_state=init_state)
                return encoder_state      # [None, sequence_len, num_units], [None, num_units]


    def _decoder_train(self):
        # decoder during training
        with tf.variable_scope("decoder"):
            self.decoder_input = tf.concat([tf.ones_like(self.input_y[:, :1]), self.input_y[:,:-1]], axis=1, name="decoder-input") # <start>
            with tf.variable_scope("embedding-layer"):
                if not self.pretrained_wordvec:
                    embedding_decoder = tf.get_variable(name="decoder-embedding",
                                                        shape=[self.vocab_size_y, self.embed_size], dtype=tf.float32)
                else:
                    embedding_decoder = np.load("chinese_wordvec.npy")
                tf.summary.histogram("embedding_decoder", embedding_decoder)
                embeded_y = tf.nn.embedding_lookup(embedding_decoder, self.decoder_input)  # [None, sequence_len, embed_size]

            with tf.variable_scope("rnn"):
                self.decoder_rnn_cell = self._get_cell(self.num_units, self.rnn_type)
                decoder_output, decoder_state = tf.nn.dynamic_rnn(cell=self.decoder_rnn_cell,
                                                                  inputs=embeded_y,
                                                                  initial_state= self.encoder_state,    # [None, num_units]
                                                                  time_major=False)
        return decoder_output  # [None, sequence_len, num_units]

    def _decoder_test(self):
        # decoding during inference/test
        decoder_output = []
        with tf.variable_scope("decoding-test"):
            decoding_input = tf.ones_like(self.input_y[:, :1] ,name="decoding")  # [None, 1]
            with tf.variable_scope("embedding-layer"):
                if not self.pretrained_wordvec:
                    embedding_decoder = tf.get_variable(name="decoder-embedding",
                                                        shape=[self.vocab_size_y, self.embed_size], dtype=tf.float32)
                else:
                    embedding_decoder = np.load("chinese_wordvec.npy")
                decoding_input = tf.nn.embedding_lookup(embedding_decoder, decoding_input)  # [None, 1, embed_size]
            with tf.variable_scope("decoding-rnn", reuse=tf.AUTO_REUSE):
                ending = tf.ones_like(decoding_input) * 2  # <end>  # [None, 1, embde_size]
                while (decoding_input != ending):
                    decoding_input, _ = tf.nn.dynamic_rnn(cell=self._get_cell(self.num_units, self.rnn_type),
                                                          inputs=decoding_input,
                                                          time_major=False,
                                                          initial_state=self.encoder_state)
                    decoder_output.append(decoding_input)
                decoder_output = tf.concat(decoding_input, axis=1) # [None, <= sequence_len, num_units]
        return decoder_output

    def _train_op(self):
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)
        return train_op

    @staticmethod
    def _get_cell(num_units,rnn_type):
        if rnn_type == "LSTM":
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=num_units)
        if rnn_type == "RNN":
            return tf.nn.rnn_cell.BasicRNNCell(num_units=num_units)
        if rnn_type == "GRU":
            return tf.nn.rnn_cell.GRUCell(num_units=num_units)