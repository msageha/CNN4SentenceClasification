import os

import numpy as np
import gensim
import MeCab

class Word2Vec():
    def __init__(self, w2v_path):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)
        self.words = self.model.vocab.keys()

    def __call__(self, word):
        if word in self.words:
            return self.model[word]
        else:
            return np.zeros(200, dtype=np.float32)

class LoadText():
    def __init__(self, path='./data', max_length=12172):
        self.class_num = len(os.listdir(path))
        self.tagger = MeCab.Tagger('')
        self.tagger.parse('')
        self.path = path
        self.max_length = max_length

    def separation(self, sentence):
        words = []
        node = self.tagger.parseToNode(sentence).next
        while node.next:
            words.append(node.surface)
            node = node.next
        return words

    def load(self, w2v):
        data_dict = {'words':[], 'class':[], 'file':[], 'vectors':[]}
        for directory in os.listdir(self.path):
            for file in os.listdir(f'{self.path}/{directory}'):
                with open(f'{self.path}/{directory}/{file}') as f:
                    lines = f.readlines()
                words = []
                for line in lines:
                    words += self.separation(line)
                vectors = []
                for word in words:
                    vectors.append(w2v(word))
                data_dict['words'].append(words)
                data_dict['class'].append(directory)
                data_dict['file'].append(file)
                # vectors = np.pad(vectors, [(0, self.max_length - len(vectors)), (0, 0)], 'constant')
                data_dict['vectors'].append(vectors)
        return data_dict
