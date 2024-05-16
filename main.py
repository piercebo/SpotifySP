import gensim
import torch
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

MAX_LENGTH_SONG = 3577 #length of the song with the longest string in apiAndLyric.csv



def buildSentenceVector(string, selectedModel="2"):
    model = None
    if selectedModel == "1":
        model = gensim.models.keyedvectors.KeyedVectors.load("./models/gloveModel300.model")
        splitText = gensim.utils.simple_preprocess(string)
        textVectors = []
        skippedWords = 0
        for word in splitText:
            try:
                vector = torch.tensor(model.get_vector(word, norm=False))
                textVectors.append(vector)
            except KeyError:
                # print("given key doesn’t exist: " + word  +". Skipping word.")
                skippedWords += 1
        if len(textVectors) == 0:
            return torch.tensor([])
        tensorStack = torch.stack(textVectors)
        averageVector = torch.sum(tensorStack, dim=0)
        averageVector = averageVector/(len(textVectors)-skippedWords)
        return averageVector
    else:
        model = gensim.models.word2vec.Word2Vec.load("./models/customModel.model")
        splitText = gensim.utils.simple_preprocess(string)
        textVectors = []
        skippedWords = 0
        for word in splitText:
            try:
                vector = torch.tensor(model.wv.get_vector(word, norm=False))
                textVectors.append(vector)
            except KeyError:
                # print("given key doesn’t exist: " + word  +". Skipping word.")
                skippedWords += 1
        if len(textVectors) == 0:
            return torch.tensor([])
        tensorStack = torch.stack(textVectors)
        averageVector = torch.sum(tensorStack, dim=0)
        averageVector = averageVector/(len(textVectors)-skippedWords)
        return averageVector

#needs to be updated
def vectorizeSongs300d():
    df = pd.read_csv("./datasets/FilteredLyrics1.csv")
    df = df.drop(["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "language"], axis=1)
    df["sentence_vector"] = df["text"].apply(buildSentenceVector)
    df.to_csv("./datasets/VectoredSongs.csv", index=False)

def vectorizeSongs100d():
    df = pd.read_csv("./datasets/FilteredLyrics1.csv")
    df = df.drop(["danceability", "energy", "key", "loudness", "mode", "speechiness", "acousticness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "language"], axis=1)
    df["sentence_vector"] = df["text"].apply(lambda text: buildSentenceVector(text))
    df.to_csv("./datasets/VectoredSongs100d.csv", index=False)

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

def MSLSTMPredict(text):
    model = tf.keras.models.load_model("./models/MultiSentLSTM_0.5.keras")
    tokenizer = Tokenizer()
    tokenizer.texts_to_sequences([text])
    sequences = tokenizer.texts_to_sequences([text])
    maxLengthSequence = max(len(sequence) for sequence in sequences)
    paddedSequences = pad_sequences(sequences, MAX_LENGTH_SONG)
    prediction = model.predict(paddedSequences)
    print("Music Inference:" + '\n' + "Danceability - " + str(prediction[0][0]) + "\nEnergy - " + str(prediction[0][1]) + "\nSpeechiness - " + str(prediction[0][2]) + "\nValence - " + str(prediction[0][3]) + '\n')
    return prediction[0]

def musicInferenceSimilarity(df, musicInference, track_id):
    impact = 10 #the percentage that music inference effects playlist decisions
    currentSong = df.loc[df['track_id'] == track_id]
    if currentSong.empty:
        return 0
    apiVector = [float(currentSong.iloc[0, 6]), float(currentSong.iloc[0, 7]), float(currentSong.iloc[0, 8]), float(currentSong.iloc[0, 9])]
    similarity = cosineSimilarity(torch.tensor(musicInference), torch.tensor(apiVector))
    return similarity.item()/impact



def curatePlaylist(inputVector, df, musicInference):
    apiDF = pd.read_csv("./datasets/apiAndLyric.csv")
    similarityList = []    
    for i in range(len(df["sentence_vector"])):
        songTensorStr = df.loc[i, "sentence_vector"]
        removedTensorOperator = songTensorStr.split('[')[1].split(']')[0]
        songTensorList = [float(weight) for weight in removedTensorOperator.split(',')]
        songTensor = torch.tensor(songTensorList)
        cosSimilarity = cosineSimilarity(songTensor, inputVector).item()
        inferenceSimilarity = musicInferenceSimilarity(apiDF, musicInference, df.loc[i, "track_id"])
        idSimilarityTuple = (cosSimilarity + inferenceSimilarity, df["track_id"][i], df["track_name"][i], df["artists"][i])
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
    df = pd.read_csv("./datasets/VectoredSongs100d.csv")
    selectedModel = input("\nPress 1 for pretrained GloVe\nPress 2 for custom model\n")
    if selectedModel == "1":
        df = pd.read_csv("./datasets/VectoredSongs.csv")
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
    inputString = input("\nEnter playlist description:\n")
    musicInference = MSLSTMPredict(inputString)
    inputVector = buildSentenceVector(inputString, selectedModel)
    if len(inputVector) == 0:
        print("\nInvalid entry. Recheck spelling.\n")
    print('\nMaking Playlist:')
    playlist = curatePlaylist(inputVector, df, musicInference)
    print("\n", playlist, "\n")

main()