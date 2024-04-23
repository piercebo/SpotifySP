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

def wordLookup(string):
    model = gensim.models.keyedvectors.KeyedVectors.load("./models/gloveModel300.model")
    lowercase = string.lower()
    try:
        vector = model.get_vector(lowercase, norm=False)
        print("[" + string +"]" + " found")
        return vector
    except KeyError:
        print("word [" + string + "] not found")

def checkInputGenre(inputString):
    genreSimilarity = 0
    subGenreSimilarity = 0
    inputGenre = ""
    inputSubGenre = ""
    inputLowerCase = inputString.lower()
    genre, subGenre = helper.genreCollection()
    for value in genre:
        if value in inputLowerCase:
            inputGenre = value
            genreSimilarity += 0.1
        if ("non " + value) in inputLowerCase:
            genreSimilarity -= 0.2
        elif ("non-" + value) in inputLowerCase:
            genreSimilarity -= 0.2
        elif ("not " + value) in inputLowerCase:
            genreSimilarity -= 0.2
    for value in subGenre:
        if value in inputLowerCase:
            inputSubGenre = value
            subGenreSimilarity += 0.2
        if ("non " + value) in inputLowerCase:
            subGenreSimilarity -= 0.4
        elif ("non-" + value) in inputLowerCase:
            subGenreSimilarity -= 0.4
        elif ("not " + value) in inputLowerCase:
            subGenreSimilarity -= 0.4
    return (inputGenre, genreSimilarity, inputSubGenre, subGenreSimilarity)
        

def inputGenreSimilarity(df, track_id, genre, genreSimilarity, subGenre, subGenreSimilarity):
    if genre == "" and subGenre == "":
        return 0
    currentRow = df.loc[df["track_id"] == track_id]
    rowGenre = currentRow["playlist_genre"].values[0]
    rowSubGenre = currentRow["playlist_subgenre"].values[0]
    similarity = 0
    if genre == rowGenre:
        similarity += genreSimilarity
    if rowSubGenre == subGenreSimilarity:
        similarity += subGenreSimilarity
    return similarity

    

def cosineSimilarity(vector1, vector2):
    return torch.nn.functional.cosine_similarity(vector1, vector2, dim=-1)

def curatePlaylist(inputVector, inputString):
    df = pd.read_csv("./datasets/VectoredSongs.csv")
    df2 = pd.read_csv("./datasets/LyricSet2.csv")
    similarityList = []
    # check for genres in input
    inputGenreTuple = checkInputGenre(inputString)
    for i in range(len(df["sentence_vector"])):
        songTensorStr = df.loc[i, "sentence_vector"]
        removedTensorOperator = songTensorStr.split('[')[1].split(']')[0]
        songTensorList = [float(weight) for weight in removedTensorOperator.split(',')]
        songTensor = torch.tensor(songTensorList)
        ############################################
        cosSimilarity = cosineSimilarity(songTensor, inputVector).item()
        genreSimilarity = inputGenreSimilarity(df2, df["track_id"][i], inputGenreTuple[0], inputGenreTuple[1], inputGenreTuple[2], inputGenreTuple[3])
        totalSimilarity = cosSimilarity+genreSimilarity
        ############################################
        idSimilarityTuple = (totalSimilarity, df["track_id"][i], df["track_name"][i], df["artists"][i])
        similarityList.append(idSimilarityTuple)
    similarityList = sorted(similarityList, key=lambda tup: tup[0], reverse=True)
    return similarityList[:11]

def main():
    inputString = input("\nEnter playlist description:\n")
    inputVector = buildSentenceVector(inputString)
    if len(inputVector) == 0:
        print("\nInvalid entry. Recheck spelling.\n")
    playlist = curatePlaylist(inputVector, inputString)
    print("\n", playlist, "\n")
    
main()