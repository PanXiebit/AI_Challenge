import nltk
import re
import jieba
import tqdm
from gensim.models import Word2Vec
import pickle
import numpy as np
import os
from preprocessing import TextPreProcessing

### 英文文本正则化预处理
# def simple_token(sent):
#     sent = re.sub(r"(\w)'s\b", r"\1 is", sent)    # make 's a separate word, and then stop it
#     sent = re.sub(r"(\w[’'])(\w)", r"\1 \2", sent) # other ' in a word creates 2 words
#     sent = re.sub(r"([\"().,;:/_?!-])", r" \1 ", sent).replace('-',' ')    # add spaces around punctuation
#     sent = re.sub(r"  *", r" ", sent)               # replace multiple spaces with just one
#     return sent.lower().split()

class MySentence():
    def __init__(self, dirname):
        self.dirname = dirname
        
    def __iter__(self):
        for line in tqdm.tqdm(open(self.dirname)):
            yield TextPreProcessing.stem(line.strip())
            
class MyZhSentence():
    def __init__(self, dirname):
        self.dirname = dirname
    
    def __iter__(self):
        for line in tqdm.tqdm(open(self.dirname)):
            yield list(jieba.cut(line.strip()))
            
def train_wordvec(lan="en", embedding_dim = 256):
    if lan == 'en':
        corpus = MySentence('train/train.en')
    else:
        corpus = MyZhSentence("train/train.zh")
    model = Word2Vec(corpus, size=embedding_dim, min_count=10, sg=1, workers=8, hs=0, window=5, iter=5)
    print(lan + "_model trained")
    
    # 词表
    vocabulary = model.wv.vocab  # dict
    words = list(vocabulary.keys())
    
    # 开头、结尾、padding、未知词
    function_words = ['<pading>','<start>','<end>','<unk>']
    words = function_words + words    
    word2index = dict(zip(words, range(len(words))))
    
    input_dim = len(words)
    embeddings = []
    # 训练好的词向量
    for word in vocabulary:
        embeddings.append(model[word])
    embeddings = np.array(embeddings, dtype=np.float32)
    
    weights = np.zeros((input_dim, embedding_dim), np.float32)
    weights[1] = np.ones(embedding_dim, np.float32) * 0.33   # start
    weights[2] = np.ones(embedding_dim, np.float32) * 0.66   # end
    weights[3] = np.average(embeddings, axis = 0)
    weights[4:] = embeddings
    
    # 保存词表
    path = "output/" + lan
    if not os.path.exists(path + "_vocab.pkl"):
        os.mknod(path + "_vocab.pkl")
        pickle.dump(word2index, open(path + '_vocab.pkl', 'wb'))
    # 保存词向量
    np.save(path + '_word2vec.npy', weights)
    print(lan + '_word vector saved!')


if __name__=="__main__":
    #train_wordvec("en")
    train_wordvec("zh")

    ### So这是一个无监督学习，怎么评价这个词向量的好坏呢？

    