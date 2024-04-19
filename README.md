# SpotifySP
 
Outside Datasets:
Spotify API slice - https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download
Lyric dataset 1 - https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset
Lyric dataset 2 - https://www.kaggle.com/datasets/imuhammad/audio-features-and-lyrics-of-spotify-songs

helper.LyricToCSVFilter() combines the spotify api data with the lyric dataset.  It produces a csv in the datasets directory called "LyricsAndAPI.csv".
If you are using a CSV highlight tool, the CSV may appear to have broken columns and rows due to commas placed within quotes; The pandas dataframe object will compile it correctly.

helper.dolchFilter() removes 79 of the most common English words.  In order to make a word embedding that embodies the true meaning of a song you must identify words that carry the most symbolism.  For our purposes and for a faster and more accurate algotithm, the dolchFilter() function removes the most common meaningless words from our text source.