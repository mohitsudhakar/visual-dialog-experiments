import numpy as np
import torch

def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel)," words loaded!")
    return gloveModel

def get_pretrained_weights(vocab):
    word_vectors = loadGloveModel('./data/glove.6B.300d.txt')
    #print("Vocab:", vocab.word2index, "len", len(vocab))
    matrix_len = len(vocab)
    weights = np.zeros((matrix_len, 300), dtype=np.float32)

    for i, word in enumerate(vocab.word2index):
        try:
            weights[i] = word_vectors[word]
        except:
            weights[i] = np.random.normal(scale = 0.6, size = (300,))

    weights = torch.tensor(weights)
    return weights