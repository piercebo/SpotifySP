import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import pandas as pd
import gensim
import torchtext.data
import torchtext.vocab



#hyperparameters
batchSize = 100
hiddenSize = 128
numLayers = 2
numEpochs = 5
learningRate = 0.01


def convertGloveModel():
    gloveModel = gensim.models.keyedvectors.KeyedVectors.load("./models/gloveModel300.model")
    embedding = []
    for word in gloveModel.index_to_key:
        embedding.append(gloveModel[word])
    embedding = torch.tensor(embedding)
    return embedding


class LSTM(nn.Module):
    def __init__(self, inputLength, hiddenSize, numLayers, embedding):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.lstm = nn.LSTM(inputLength, hiddenSize, numLayers, batch_first=True)        

    def forward(self, x):
        embedding = self.embedding(x)
        embedding = torch.squeeze(embedding)
        output, _ = self.lstm(embedding)
        return output


def tokenize():
    df = pd.read_csv("./datasets/BrownProcessed.csv")
    tokenizer = torchtext.data.utils.get_tokenizer("basic_english")
    tokenized = [tokenizer(text) for text in df["tokenized_text"]]
    vocabulary = torchtext.vocab.build_vocab_from_iterator(tokenized)
    indexedTokens = [[vocabulary[token] for token in tokens] for tokens in tokenized]
    maxLengthToken = max(len(tokens) for tokens in indexedTokens)
    paddedTokens = [tokens + [0] * (maxLengthToken - len(tokens)) for tokens in indexedTokens]
    dataset = torch.utils.data.TensorDataset(torch.tensor(paddedTokens, dtype=torch.long))
    dataLoader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)
    return dataLoader

def trainModel():
    embeddings = convertGloveModel()
    inputLength = embeddings.shape[1]
    model = LSTM(inputLength, hiddenSize, numLayers, embeddings)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    dataLoader = tokenize()
    for epoch in range(numEpochs):
        model.train()
        totalLoss = 0
        for input in dataLoader:
            optimizer.zero_grad()
            inputData = torch.stack(input, dim=0)
            outputs = model(inputData)
            loss = criterion(outputs, inputData)
            loss.backward()
            optimizer.step()
            totalLoss += loss.item()
        print(f'Epoch {epoch+1}/{numEpochs}, Loss: {totalLoss/len(dataLoader)}')
    torch.save(model.state_dict(), './models/lstmModel.pth')

trainModel()