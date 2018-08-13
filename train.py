import tensorflow as tf
from model.transformer import Transformer
import os
from sent2id import build_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# prepare  train dataset
if not os.path.exists("output/en_dataset.npy"):
    print("prepare en dataset!")
    build_dataset()
if not os.path.exists("output/zh_dataset.npy"):
    print("prepare zh dataset")
    build_dataset(lan="zh")
en_train_data = np.squeeze(np.load("output/en_dataset.npy"),axis=1) # [None, 1, 30]->[None,30]   (9903244,30)
zh_train_data = np.squeeze(np.load("output/zh_dataset.npy"),axis=1)  # [None, 1, 30]->[None, 30] (9903244,30)
assert (len(en_train_data) == len(zh_train_data))

train_en, valid_en, train_zh, valid_zh = train_test_split(en_train_data, zh_train_data, test_size=0.2)
# train:[7922595, 30]  valid:[1980649, 30]

# pretrained wordvec
en_embedding = np.load("output/en_word2vec.npy") # (66891, 256)
zh_embedding = np.load("output/zh_word2vec.npy") # (143221, 256)


# Hyperparameters
flags = tf.app.flags
flags.DEFINE_integer("sentence_len", 30, "the max length of a sentence")
flags.DEFINE_float("learning_rate", 0.01, "the learning rate")
flags.DEFINE_float("keep_prob", 0.1, "the probability of dropout to keep")
flags.DEFINE_integer("embed_size", 256, "the dimension size of english and chinese word vector")
flags.DEFINE_integer("batch_size", 60, "the size of one batch")
flags.DEFINE_integer("num_epochs", 10, "the number of epochs")
FLAGS = flags.FLAGS


assert (FLAGS.sentence_len == en_train_data.shape[1])
assert (FLAGS.embed_size == en_embedding.shape[1])

VOCAB_SIZE_EN = en_embedding.shape[0]
VOCAB_SIZE_CN = zh_embedding.shape[0]

# iter to get the train dataset
def batch_iter(train_en, train_zh, batch_size, num_epochs):
    num_batches_per_epoch = (len(train_en) - 1) // batch_size + 1  # 7922595//128=61895 这会不会太多了啊?????但是batch太大,内存不够吧?
    for i in range(num_epochs):
        index = np.arange(len(train_en))
        np.random.shuffle(index)
        train_en = train_en[index]
        train_zh = train_zh[index]
        for batch_num in range(num_batches_per_epoch):
            start = batch_num * batch_size
            end = min(start + batch_size, len(train_en))
            yield train_en[start:end], train_zh[start:end]

model = Transformer(d_k=32,
                    d_v=32,
                    d_model=256,
                    sentence_len=FLAGS.sentence_len,
                    vocab_size_cn=VOCAB_SIZE_CN,
                    vocab_size_en=VOCAB_SIZE_EN,
                    heads=8,
                    num_layers=6,
                    learning_rate=FLAGS.learning_rate,
                    dropout_keep_pro=FLAGS.keep_prob)

saver = tf.train.Saver()
merged = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./temp/graph", graph=sess.graph)
    batch_data = batch_iter(train_en, train_zh, FLAGS.batch_size, FLAGS.num_epochs)
    num_batches_per_epoch = (len(train_en) - 1) // FLAGS.batch_size + 1
    max_accuracy = 0
    for i, (en_batch, zh_batch) in enumerate(batch_data):
        train_feed = {model.input_x:zh_batch,
                      model.input_y:en_batch,
                      model.is_training:True}
        summary, _, step, acc_, loss_ = sess.run([merged, model.train_op, model.global_step, model.loss, model.acc],
                                                 feed_dict=train_feed)
        if step % 200 == 0:
            print("step {0}: loss = {1}, acc:{2}".format(step, loss_, acc_))
            writer.add_summary(summary, step)

        if step % 1000 == 0:
            valid_batch = batch_iter(valid_en, valid_zh, 1000, 1)
            val_accuracy, cnt = 0, 0
            for valid_en_batch, valid_zh_batch in valid_batch:
                valid_feed_dict = {
                    model.input_x:valid_zh_batch,
                    model.input_y:valid_en_batch,
                    model.is_training:False}

                accuracy = sess.run(model.acc, feed_dict=valid_feed_dict)
                val_accuracy += accuracy
                cnt += 1

            val_accuracy = val_accuracy / cnt
            print("\n{0} steps, Validation Accuracy is {1}\n".format(step // num_batches_per_epoch, val_accuracy / cnt))
    saver.save(sess, "temp/model/save")
writer.close()