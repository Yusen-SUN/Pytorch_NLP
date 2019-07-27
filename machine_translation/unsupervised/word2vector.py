import unicodedata
import string
import re



class Vocab:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {"PAD":0, "STYLE_1":1, "STYLE_2":2, "EOS":3}
        self.word2count = {}
        self.index2word = {0:"PAD", 1:"STYLE_1", 2:"STYLE_2",  3:"EOS"}
        self.num_vocab = 4
        
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
            
# def unicode2Ascii(s):
#     return ''.join(
#     c for c in unicodedata.normalize('NFD', s)
#     if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = s.lower().strip()
    #s = unicode2Ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(vocab, sentence, length, style):
    
    PAD = 0
    STYLE_1 = 1
    STYLE_2 = 2
    EOS = 3
    
    words = sentence.split(' ')
    num_words = len(words)
    seq = []
    sentence_length = 0
    
    # only choose sentence under certain length
    if num_words+2<length:
        sentence_length = num_words+2
        if style==STYLE_1:
            seq.append(STYLE_1)
        else:
            seq.append(STYLE_2)
        for w in words:
            seq.append(vocab.word2index[w])
        seq.append(EOS)
        for i in range(num_words+1, length):
            seq.append(PAD)
        
#     else:
#         sentence_length = length
#         for i in range(length-1):
#             seq.append(vocab.word2index[words[i]])
#         seq.append(EOS)
  
    return seq, sentence_length

def sentenceFromIndex(vocab, seq):
    return [vocab.index2word[s] for s in seq]


# def sentencesFromTensor(vocab, tensors):
#     for i in tensors:
#         print(' '.join(word_processing.sentenceFromIndex(vocab, i.cpu().numpy())))