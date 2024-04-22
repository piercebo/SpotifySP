import helper
import gensim
import torch
import pandas as pd

def buildSentenceVector(string):
    model = gensim.models.keyedvectors.KeyedVectors.load("./models/gloveModel300.model")
    splitText = gensim.utils.simple_preprocess(string)
    textVectors = []
    skippedWords = 0
    for word in splitText:
        try:
            vector = torch.tensor(model.get_vector(word, norm=False))
            textVectors.append(vector)
        except KeyError:
            print("given key doesn’t exist: " + word  +". Skipping word.")
            skippedWords += 1
    if len(textVectors) == 0:
        return torch.tensor([])
    tensorStack = torch.stack(textVectors)
    averageVector = torch.sum(tensorStack, dim=0)
    averageVector = averageVector/(len(textVectors)-skippedWords)
    return averageVector

def vectorizeSongs():
    df = pd.read_csv("./datasets/FilteredLyrics1.csv")
    df = df.drop(["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "language"], axis=1)
    df['sentence_vector'] = df["text"].apply(buildSentenceVector)
    df.to_csv("./datasets/VectoredSongs.csv", index=False)

print(buildSentenceVector("hello world"))