TF_ENABLE_ONEDNN_OPTS=0
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def MultiSentInit():
    #tokenize text
    df = pd.read_csv("./datasets/apiAndLyric.csv")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df["text"])
    sequences = tokenizer.texts_to_sequences(df["text"])
    maxLengthSequence = max(len(sequence) for sequence in sequences)
    paddedSequences = pad_sequences(sequences, maxLengthSequence)
    numClasses = 4

    #prepare numpy array for categories
    categoryData = []
    for index, row in df.iterrows():
        dance = float(row["danceability"])
        energy = float(row["energy"])
        speech = float(row["speechiness"])
        valence = float(row["valence"])
        categoryArray = [dance, energy, speech, valence]
        categoryData.append(categoryArray)
    categoryData = np.array(categoryData)     

    #split data
    xTrain, xTest, yTrain, yTest = train_test_split(paddedSequences, categoryData, test_size=0.2, random_state=2147482647)
    epochs = 10
    embeddingDim = 300
    batchSize = 256

    #build model
    model = Sequential()
    model.add(Embedding(input_dim=len(tokenizer.word_index), output_dim=embeddingDim, input_length=paddedSequences.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dense(4, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
    
    #train model and save
    history = model.fit(xTrain, yTrain, epochs=epochs, batch_size=batchSize, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=7, min_delta=0.0001)])
    model.save_weights("./models/MultiSentLSTM")

    #graph accuracy and loss
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()
