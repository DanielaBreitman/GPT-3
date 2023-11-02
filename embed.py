import torch.nn as nn
import torch

class WordEmbedder(object):
    def __init__(self, word_dictionary, d_hidden):
        self.word_dictionary = word_dictionary
        self.d_hidden = d_hidden

        self.embed_layer = nn.Linear(len(word_dictionary), d_hidden)
        self.unembed_layer = nn.Linear(d_hidden, len(word_dictionary))

    def embed(self, tokens):
        d_t = len(tokens)
        tokens = [self.word_dictionary.to_num(t) for t in tokens]

        one_hot = torch.zeros((d_t, len(self.word_dictionary)))

        for t in range(d_t):
            one_hot[t][tokens[t]] = 1

        return self.embed_layer(one_hot)
    
    def unembed(self, X):
        # X (d_t, d_model)
        d_t = X.shape[0]

        one_hot = self.unembed_layer(X)

        res = torch.argmax(one_hot, axis=1)
        res = [self.word_dictionary.to_word(n) for n in res]

        return res

class WordDictionary(object):
    def __init__(self):
        self.word2num = dict()
        self.num2word = dict()
        self.count = 0

    def add(self, word):
        if word not in self.word2num:
            self.word2num[word] = self.count
            self.num2word[self.count] = word
            self.count+1
    
    def to_num(self, word):
        return self.word2num[word]
    
    def to_word(self, num):
        return self.num2word[num]

    def __len__(self):
        return self.count