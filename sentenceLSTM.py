import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential

def initLSTM():
    df = pd.read_csv("./datasets/BrownProcessed.csv")
    df["tokenized_text"] = df["tokenized_text"].apply(str.lower)
    vocabSize = 20000
    hotEncoding = [one_hot(sentence, vocabSize)for sentence in df["tokenized_text"]] 
    paddingLength = max(len(encoding) for encoding in hotEncoding)
    paddedEmbeddings = pad_sequences(hotEncoding, padding="pre", maxlen=paddingLength)
    features = 300
    model = Sequential()
    model.add(Embedding(vocabSize, features, input_length=paddingLength))
    model.compile("adam", "mse")
    print(model.predict(paddedEmbeddings[0]))
    model.save('./models/kerasLSTM.keras')

def loadLSTM():
    return tf.keras.models.load_model('./models/kerasLSTM.keras')

def predictLSTM(string):
    model = loadLSTM()
    print(model.predict(string))

initLSTM()
