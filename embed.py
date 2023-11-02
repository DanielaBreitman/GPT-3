import torch.nn as nn
import torch

class SkipGram(object):
    def __init__(self, word_dict, d_hidden):
        self.word_dict = word_dict
        self.d_hidden = d_hidden

        self.embed_layer = nn.Linear(len(word_dict), d_hidden)
        self.unembed_layer = nn.Linear(d_hidden, len(word_dict))
        self.softmax = nn.Softmax(len(word_dict))

    def to_vector(self, words):
        context_vector = self._to_context_vector(words)
        return self.embed_layer(context_vector)
    
    def to_word(self, pred):
        idx = torch.argmax(pred)
        return self.word_dict.to_word(idx)

    def forward(self, context_vector):
        hidden_layer = self.embed_layer(context_vector)
        output = self.unembed_layer(hidden_layer)
        return self.softmax(output)
    
    def _to_context_vector(self, words):
        context_vector = torch.zeros(len(self.word_dict))
        for w in words:
            num = self.word_dict.to_num(w)
            context_vector[num] = 1
        return context_vector



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