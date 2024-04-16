import pandas as pd

def LyricToCSVFilter(APISlice, lyricSet):
    df1 = pd.read_csv(APISlice)
    df1 = df1.drop(["index"], axis=1)
    df2 = pd.read_csv(lyricSet)
    df2 = df2.drop(["link"], axis=1)
    df1 = df1.merge(df2, how="inner", on=["artists", "track_name"])
    df1.to_csv("./datasets/LyricsAndAPI.csv", index=False)
    


LyricToCSVFilter("./datasets/SpotifyAPISlice.csv", "./datasets/LyricSet.csv")