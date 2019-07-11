#import unicodedata
import string
import re

class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD":0, "SOS":1, "EOS":2}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.num_vocab = 3
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_vocab
            self.word2count[word]= 1
            self.index2word[self.num_vocab] = word
            self.num_vocab += 1
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)
            
    
    def trim(self, min_count):
        if self.trimmed:
            return
        
        keep_words = []

        for k, v in self.word2count.items():
            if v >= self.lower_bound:
                keep_words.append(k)
        
        print('keep words {} / {} = {:.4f}'.format(
            len(keep_words), len(self.word2index), len(keep_words) / len(word2index)
        ))
        
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"SOS", 2:"EOS"}
        self.num_vocab = 3
        for word in keep_words:
            self.addWord(word)
            
# def unicode2Ascii(s):
#     return ''.join(
#     c for c in unicodedata.normalize('NFD', s)
#     if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    #s = unicode2Ascii(s.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    s = s.lower().strip()
    s = re.sub(r"[^a-zA-Z]+", r" ", s)
    return s

def indexesFromSentence(vocab, sentence, length):
    
    PAD = 0
    SOS = 1
    EOS = 2
    
    words = sentence.split(' ')
    num_words = len(words)
    seq = []
    sentence_length = 0
    
    if num_words<length:
        sentence_length = num_words+1
        for w in words:
            seq.append(vocab.word2index[w])
        seq.append(EOS)
        for i in range(num_words+1, length):
            seq.append(PAD)
        
    else:
        sentence_length = length
        for i in range(length-1):
            seq.append(vocab.word2index[words[i]])
        seq.append(EOS)
  
    return seq

def sentenceFromIndex(vocab, seq):
    return [vocab.index2word[s] for s in seq]


def sentencesFromTensor(vocab, tensors):
    for i in tensors:
        print(' '.join(word_processing.sentenceFromIndex(vocab, i.cpu().numpy())))