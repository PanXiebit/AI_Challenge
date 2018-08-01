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
en_train_data = np.squeeze(np.load("output/en_dataset.npy"),axis=1) # [None, 1, 30]->[None,30]
zh_train_data = np.squeeze(np.load("output/zh_dataset.npy"),axis=1)  # [None, 1, 30]->[None, 30]
assert (len(en_train_data) == len(zh_train_data))

train_en, valid_en, train_zh, valid_zh = train_test_split(en_train_data, zh_train_data, test_size=0.2)
# train:[7922595, 30]  valid:[1980649, 30]

# pretrained wordvec
en_embedding = np.load("output/en_word2vec.npy") # (66891, 256)
zh_embedding = np.load("output/zh_word2vec.npy") # (143221, 256)


# Hyperparameters
SENTENCE_LEN = en_train_data.shape[1]  # 30
EMBED_SIZE = en_embedding.shape[1]     # 300
VOCAB_SIZE_EN = en_embedding.shape[0]
VOCAB_SIZE_CN = zh_embedding.shape[0]
BATCH_SIZE = 128
NUM_EPOCH = 5

# iter to get the train dataset
def batch_iter(train_en, train_zh):
    index = np.arange(len(train_en))
    np.random.shuffle(index)
    train_en = train_en[index]
    train_zh = train_zh[index]

    num_batches_per_epoch = (len(train_en) - 1) // BATCH_SIZE + 1  # 7922595/128=61895 这会不会太多了啊?????但是batch太大,内存不够吧?
    for i in range(NUM_EPOCH):
        for batch_num in range(num_batches_per_epoch):
            start = batch_num * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(train_en))
            yield train_en[start:end], train_zh[start:end]

model = Transformer(d_k=32,
                    d_v=32,
                    d_model=256,
                    sentence_len=SENTENCE_LEN,
                    vocab_size_cn=VOCAB_SIZE_CN,
                    vocab_size_en=VOCAB_SIZE_EN,
                    heads=8,
                    num_layers=6,
                    learning_rate=0.01,
                    dropout_keep_pro=0.5,
                    initializer=tf.random_normal_initializer())


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_data = batch_iter(train_en, train_zh)
    num_batches_per_epoch = (len(train_en) - 1) // BATCH_SIZE + 1
    max_accuracy = 0
    for en_batch, zh_batch in batch_data:
        train_feed = {model.input_x:en_batch,
                      model.input_y:zh_batch,
                      model.is_training:True}
        _, step, acc_, loss_ = sess.run([model.train_op, model.global_step, model.loss, model.acc],
                                feed_dict=train_feed)
        if step % 200 == 0:
            print("step {0}: loss = {1}".format(step, loss_))

        if step % 2000 == 0:
            valid_batch = batch_iter(valid_en, valid_zh)
            sum_accuracy, cnt = 0, 0
            for valid_en_batch, valid_zh_batch in valid_batch:
                valid_feed_dict = {
                    model.input_x:valid_en_batch,
                    model.input_y:valid_zh_batch,
                    model.is_training:False}

                accuracy = sess.run(model.acc, feed_dict=valid_feed_dict)
                sum_accuracy += accuracy
                cnt += 1

            valid_accuracy = sum_accuracy / cnt
            print("\n{0} steps, Validation Accuracy is {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))

