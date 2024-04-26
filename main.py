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

def cosineSimilarity(vector1, vector2):
    return torch.nn.functional.cosine_similarity(vector1, vector2, dim=-1)

def curatePlaylist(inputVector, df):
    similarityList = []    
    for i in range(len(df["sentence_vector"])):
        songTensorStr = df.loc[i, "sentence_vector"]
        removedTensorOperator = songTensorStr.split('[')[1].split(']')[0]
        songTensorList = [float(weight) for weight in removedTensorOperator.split(',')]
        songTensor = torch.tensor(songTensorList)
        cosSimilarity = cosineSimilarity(songTensor, inputVector).item()
        idSimilarityTuple = (cosSimilarity, df["track_id"][i], df["track_name"][i], df["artists"][i])
        similarityList.append(idSimilarityTuple)
    uniquetup = set(tuple())
    slist = []
    for i in similarityList:
        newtup = tuple((i[0],i[2],i[3]))
        if newtup not in uniquetup:
            tup = (i[0],i[2],i[3])
            slist.append(tup)
            uniquetup.add(newtup)
    similarityList = sorted(slist, key=lambda tup: tup[0], reverse=True)
    return similarityList[:11]

def main():
    df = pd.read_csv("./datasets/VectoredSongs.csv")
    inputString = input("\nEnter playlist description:\n")
    genres = ['rock', 'r&b', 'pop', 'edm', 'latin', 'rap']
    subGenres = ['classic rock', 'hard rock', 'new jack swing', 'neo soul', 'dance pop', 'urban contemporary', 'big room', 'hip pop', 'latin pop', 'indie poptimism', 'gangster rap', 'album rock', 'post-teen pop', 'trap', 'latin hip hop', 'southern hip hop', 'tropical', 'electropop', 'progressive electro house', 'pop edm', 'reggaeton', 'hip hop', 'permanent wave', 'electro house']
    print('\n', genres)
    genreInput = input("Enter playlist genre (press enter to skip, type \'subgenres\' for subgenres):\n")
    if genreInput in genres:
        df = df[df["playlist_genre"] == genreInput].reset_index(drop=True)
    elif genreInput == "subgenres":
        print('\n', subGenres)
        subGenreInput = input("Enter playlist subgenre (press enter to skip):\n")
        if subGenreInput in subGenres:
            df = df[df["playlist_subgenre"] == subGenreInput].reset_index(drop=True)
    inputVector = buildSentenceVector(inputString)
    if len(inputVector) == 0:
        print("\nInvalid entry. Recheck spelling.\n")
    print('\nMaking Playlist:')
    playlist = curatePlaylist(inputVector, df)
    print("\n", playlist, "\n")
    
main()