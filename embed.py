class WordDictionary(object):
    def __init__(self):
        self.mapping = dict()
        self.count = 0

    def add(self, word):
        if word not in self.mapping:
            self.count += 1
            self.mapping[word] = self.count

    def size(self):
        return self.count

    def __len__(self):
        return self.count