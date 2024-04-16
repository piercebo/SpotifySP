# SpotifySP
 
Outside Datasets:
Spotify API slice - https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset?resource=download
Lyric dataset - https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

helper.LyricToCSVFilter() combines the spotify api data with the lyric dataset.  It produces a csv in the datasets directory called "LyricsAndAPI.csv".
If you are using a CSV highlight tool, the CSV may appear to have broken columns and rows due to commas placed within quotes; The pandas dataframe object will compile it correctly.