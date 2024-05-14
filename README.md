# SpotifySP

Requires the following libraries: <br />
PyTorch - ``` pip3 install torch torchvision torchaudio ``` (CPU version) <br />
Pandas - ``` pip install pandas ``` <br />
Gensim - ``` pip install --upgrade gensim ``` <br />


Outside Datasets: <br />
Spotify API slice 1 - https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download <br />
Spotify API slice 2 - https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset <br />
Spotify API slice 3 - https://www.kaggle.com/datasets/theoverman/the-spotify-hit-predictor-dataset?select=dataset-of-10s.csv <br />
Lyric dataset 1 - https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset <br />
Lyric dataset 2 - https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs <br />
NLP Training set1 - https://www.kaggle.com/datasets/stanfordu/stanford-natural-language-inference-corpus?select=snli_1.0_train.csv <br />
Brown Corpus - https://www.kaggle.com/datasets/nltkdata/brown-corpus?select=brown.csv <br />
NegEx Phrases - https://code.google.com/archive/p/negex/downloads <br />
LSTM Helper Notebook - https://www.kaggle.com/code/ngyptr/multi-class-classification-with-lstm/notebook <br />

helper.LyricToCSVFilter() combines the spotify api data with the lyric dataset.  It produces a csv in the datasets directory called "LyricsAndAPI.csv".
If you are using a CSV highlight tool, the CSV may appear to have broken columns and rows due to commas placed within quotes; The pandas dataframe object will compile it correctly.

helper.dolchFilter() removes 79 of the most common English words.  In order to make a word embedding that embodies the true meaning of a song you must identify words that carry the most symbolism.  For our purposes and for a faster and more accurate algotithm, the dolchFilter() function removes the most common meaningless words from our text source.

Running the helper.py file will currently create the word vector model using the GloVe Wikipedia + Gigaword 5 300d vector dataset provided through Gensim.  It is approximatly 400 MB.  Because of that, .npy files are in the .gitignore file.  helper.gloveModelTest() provides some insight into how the models can be used.  I recommend watching this video (part 2 of 2) if you want someone to walk through it with you more: https://www.youtube.com/watch?v=Q2NtCcqmIww&t=6s

main.buildSentenceVector() takes the average of all the word vectors inside a sentence.  That means we have a vector representation of the general meaning of a sentence.  In datasets/VectoredSongs.csv, all the songs have a generalized sentence embedding using this method.  Not every word in every song has a word embedding in the GloVe model.  These are words that aren't actual words ('gankin', 'bankin', 'heeeeeeeeeeeyyyyyyyy'), most expletive words, and non-english words (알아줬으면), despite all the songs in the datasets/FilteredLyrics1.csv being categorized as english.

main.vectorizeSongs() is the function that created the datasets/VectoredSongs.csv. I forgot to set my tensor device as my GPU and don't know if the PyTorch library automatically chooses to use the CPU or GPU given the kind of PyTorch that you have installed.  If you have your tensor device set to an Nvidia GPU with CUDA drivers this function might run faster than what it did for me.  Runnning this function takes approximately an hour.

Running main.py will execute the main.curatePlaylist() function which will analyze the given prompt and compare its sentence vector with the vectors of our songs.  The main function will print a 10 song playlist that best matches the given prompt.  The gloveModel300 Gensim neural network must be in your models directory for this current version to run. Calling gensimGloveInit() in the helper.py file will create it for you.