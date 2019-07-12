import numpy as np


class pre_embedding:
    def __init__(self, name):
        # glove or fasttext
        self.name = name
        self.word2index = {"PAD":0, "SOS":1, "EOS":2}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.embedding = []
        
        
    def init(self):
        
        for j in range(3):
            self.embedding.append(np.random.randn(300))
        
        if self.name == 'glove':
            
            with open('./data/style_transfer/glove_vectors.txt', 'r') as v:
                whole = v.read()
            word_vector = whole.split('\n')
            word = []
            vector = []
            
            for i in range(len(word_vector)):
                if not word_vector[i]:
                    continue
                temp = word_vector[i].split()
                
                self.word2index[temp[0]] = i+3
                self.index2word[i+3] = temp[0]
                self.embedding.append(np.asarray([float(i) for i in temp[1:]]))
                
            self.embedding = np.asarray(self.embedding)
            
    def sentence2embedding(self, sentence, max_length):
        
        seq, length = self.sentence2seq(sentence, max_length)
        
        embedding = []
        
        for index in seq:
            
            embedding.append(self.embedding[index])
            
        embedding = np.asarray(embedding)
        
        return embedding, length
    
    
    def sentence2seq(self, sentence, max_length, i):
        
        words = sentence.split() + ['EOS']
        length = len(words) - 1
        
        if length<=0:
            print(i,words)
        
        if len(words) < max_length:
            words += ['PAD']*(max_length - len(words))
            
        else:
            words = words[:max_length]
            length = max_length
            
        
        seq = []
        for word in words:
            if word in self.word2index:
                seq.append(self.word2index[word])
            else:
                seq.append(self.word2index['<unk>'])
            
        seq = np.asarray(seq)
        
        return seq, length
    
    
    
    def seq2sentence(self, seq):
        
        sentence = []
        
        for i in seq:
            
            if i ==2:
                break
            
            sentence.append(self.index2word[i])
        
        return sentence
    

            