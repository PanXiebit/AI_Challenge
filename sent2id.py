import numpy as np
import pickle
from word_vec import MySentence, MyZhSentence
from keras.preprocessing.sequence import pad_sequences
import os
from word_vec import train_wordvec
from collections import Iterable


def build_dataset(train=True, lan="en", max_len=30):
    # 最大长度的设计有待考虑，因为这是翻译，也不是分类，有多长就要设置多长吧，那可能很长很长很长吧。。。像google翻译
    path = "output/" + lan
    if not os.path.exists(path + "_vocab.pkl"):
        print(lan + " train word2vec!")
        train_wordvec()
    word2id = pickle.load(open(path + '_vocab.pkl', 'rb'))  # dict

    if lan == "en":
        train_corpus = MySentence("train/train.en")  # iterable
    else:
        train_corpus = MyZhSentence("train/train.zh")

    dataset = []
    for sent in train_corpus:
        # 将每一个word转换成index，未知词设为 <unk>
        sent = list(map(lambda word: word2id.get(word, word2id["<unk>"]), sent))
        # 每一行的末尾加上 <end>
        sent = sent + [word2id["<end>"]]
        # 长度不足的padding
        # sent = sent[:max_len]
        # sent = sent + (max_len - len(sent)) * [word2id["<pading>"]]

        # sequences: List of lists, where each element is a sequence.
        sent = pad_sequences([sent], max_len, padding='post', truncating='post', value=word2id["<pading>"],
                             dtype=np.int32)
        dataset.append(sent)
    dataset = np.array(dataset)
    np.save(path + "_dataset.npy", dataset)
    print(lan + " dataset saved!")

if __name__ == "__main__":
    #build_dataset()
    build_dataset(lan="zh")