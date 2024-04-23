import gensim.downloader
import gensim.scripts
import gensim.scripts.glove2word2vec
import gensim.test
import gensim.test.utils
import pandas as pd
import gensim
import re

def LyricToCSVFilter(lyricSet):
    df2 = pd.read_csv(lyricSet)
    df2 = df2.drop(["track_album_id"], axis=1)
    df2 = df2.drop(["track_album_name"], axis=1)
    df2 = df2.drop(["playlist_name"], axis=1)
    df2 = df2.drop(["track_popularity"], axis=1)
    df2 = df2.drop(["track_album_release_date"], axis=1)
    df2 = df2.drop(["playlist_id"], axis=1)
    df2 = df2.drop(["playlist_genre"], axis=1)
    df2 = df2.drop(["playlist_subgenre"], axis=1)
    df2 = df2[df2['language'] == 'en']
    df2.to_csv("./datasets/LyricsAndAPI2.csv", index=False)
    
def dolchFilter(csvFile):
    df = pd.read_csv(csvFile)
    regexList = [r'\bthe\b', r'\bis\b', r'\bbe\b', r'\bto\b', r'\bof\b', r'\band\b', r'\ba\b', r'\bin\b', r'\bthat\b', r'\bhave\b', r'\bit\b', r'\bfor\b', r'\bon\b', r'\bwith\b', r'\bas\b', r'\bdo\b', r'\bat\b', r'\bthis\b',
                 r'\bbut\b', r'\bby\b', r'\bfrom\b', r'\bthey\b', r'\bwe\b', r'\bsay\b', r'\bor\b', r'\ban\b', r'\bwill\b', r'\bmy\b', r'\b\b', r'\ball\b', r'\bwould\b', r'\bthere\b', r'\btheir\b', r'\bwhat\b', r'\bso\b', r'\bup\b', r'\bout\b',
                 r'\bif\b', r'\babout\b', r'\bwho\b', r'\bget\b', r'\bwhich\b', r'\bgo\b', r'\bme\b', r'\bwhen\b', r'\bmake\b', r'\bcan\b', r'\blike\b', r'\btime\b', r'\bno\b', r'\bjust\b', r'\bknow\b', r'\btake\b', r'\binto\b', r'\bsome\b', r'\bcould\b',
                 r'\bthem\b', r'\bsee\b', r'\bother\b', r'\bthan\b', r'\bthen\b', r'\bnow\b', r'\blook\b', r'\bonly\b', r'\bcome\b', r'\bits\b', r'\bover\b', r'\bthink\b', r'\balso\b', r'\bback\b', r'\bafter\b', r'\buse\b', r'\bhow\b', r'\bour\b', r'\beven\b', r'\bany\b',
                 r'\bthese\b', r'\bmost\b', r'\bus\b'] 
    whitespaces = r'\s+'
    print(len(regexList))
    for i in range(len(df['text'])):
        for regex in regexList:
            df.loc[i, "text"] = re.sub(regex, "", df['text'][i], flags=re.IGNORECASE)
        df.loc[i, "text"] = re.sub(whitespaces, " ", df['text'][i])
    df.to_csv("./datasets/FilteredLyrics1.csv", index=False)

#corpus is list of strings or, more specifically, sentences
def gensimTrain(corpus: list):
    model = gensim.models.word2vec.Word2Vec.load("./model1.model")
    data = {"text": corpus}
    df = pd.DataFrame(data)
    print(df.shape)
    splitText = df["text"].apply(gensim.utils.simple_preprocess)
    model.build_vocab(splitText, progress_per=10000, update=True)
    model.train(splitText, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./model1.model")

def gensimGloveInit():
    # glove model 100: "glove-wiki-gigaword-100"
    # glove model 200: ".glove-wiki-gigaword-200"
    # glove model 300: "glove-wiki-gigaword-300" <- Recommended

    model = gensim.downloader.load('glove-wiki-gigaword-300')
    model.save("./models/gloveModel300.model")

def customStandfordTest(string):
    model = gensim.models.word2vec.Word2Vec.load("models/customStandford.model")
    print('\n', model.wv.most_similar(string), '\n')
    print(model.wv.similarity(w1=string, w2="hello"), '\n')

def gloveModelTest(string):
    # glove model 100: "./models/gloveModel100.model"
    # glove model 200: "./models/gloveModel200.model"
    # glove model 300: "./models/gloveModel300.model" <- Recommended

    model = gensim.models.keyedvectors.KeyedVectors.load("./models/gloveModel300.model")
    print('\n', model.most_similar(string), '\n')
    print(model.similarity(w1=string, w2="hello"), '\n')

