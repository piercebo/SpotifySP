import gensim.downloader
import gensim.scripts
import gensim.scripts.glove2word2vec
import gensim.test
import gensim.test.utils
import pandas as pd
import gensim
import numpy as np

def customTest(string):
    model = gensim.models.word2vec.Word2Vec.load("./models/customModel.model")
    print('\n', model.wv.most_similar(string), '\n')
    print(model.wv.similarity(w1=string, w2="hello"), '\n')

#corpus is list of strings or, more specifically, sentences
def gensimTrain(corpus: list):
    model = gensim.models.word2vec.Word2Vec.load("./models/customModel.model")
    data = {"text": corpus}
    df = pd.DataFrame(data)
    splitText = df["text"].apply(gensim.utils.simple_preprocess)
    model.build_vocab(splitText, progress_per=10000, update=True)
    model.train(splitText, total_examples=model.corpus_count, epochs=10)
    model.save("./models/customModel.model")

df1 = pd.read_csv("./datasets/BrownProcessed.csv")
df2 = pd.read_csv("./datasets/SnliCorpus.csv")
df3 = pd.read_csv("./datasets/NegExPhrases.csv")
