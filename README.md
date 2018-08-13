# AI_Challenge
chinese-to-english translation using tensorflow.

### Dataset
[AI challenger](https://challenger.ai/competition/translation): The English-Chinese Machine Translation track aims to challenge the state-of-the-art translation algorithms. Specifically, this competition track will focus on translating English transcripts to Chinese transcripts. For winner evaluation, the scores from machine translation objective metrics (BLEU) will be combined with scores from AI Challenger judge panel as well as contestants' final presentations.

Dataset analysis:  

||english|chinese|
|---|---|---|
|max_length|15/30||
|train corpus|9903244|9903244|
|testcorpus|||
|word dicts|||
|pretrained wordvec|GloVe_300d||
|gensim wordvec|256d|256d|

### Model
Transformer which is released by Google in the paper [Attention is all your need](https://arxiv.org/abs/1706.03762)  

### Trainging
```
python train.py --keep_prob 0.1 --batch_size 60
```

### Result
