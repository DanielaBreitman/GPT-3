import torch.nn as nn
import torch

class SkipGram(nn.Module):
    def __init__(self, word_dict, d_hidden):
        self.word_dict = word_dict
        self.d_hidden = d_hidden

        self.embed_layer = nn.Linear(len(word_dict), d_hidden)
        self.unembed_layer = nn.Linear(d_hidden, len(word_dict))
        self.softmax = nn.Softmax(len(word_dict))

    # Applied word embedding to a word
    def to_vector(self, word):
        word_vector = self._get_context_vector([word])
        return self.embed_layer(word_vector)
    
    # Apply word embedding to a list of vectors
    def to_vector_all(self, words):
        word_vectors = [self._get_context_vector([w]) for w in words]
        word_vectors = torch.tensor(word_vectors)
        return self.embed_layer(word_vectors)
    
    # Translate a predicted word_vector to a word
    def to_word(self, pred):
        idx = torch.argmax(pred)
        return self.word_dict.to_word(idx)

    # Forward propagation for training
    #   word_vector: a context_vector with only 1 word
    def forward(self, word_vector):
        hidden_layer = self.embed_layer(word_vector)
        output = self.unembed_layer(hidden_layer)
        return self.softmax(output)
    
    def calculate_error(self, pred, context_vector):
        error = (context_vector - pred) ** 2
        return torch.sum(error) / len(self.word_dict)
        
    # Translate the list of words to a context_vector
    def to_context_vector(self, words):
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