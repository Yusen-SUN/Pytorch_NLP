import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layer=2, dropout=0.1, embedding=None):
        super(EncoderRNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.hidden_dim = hidden_dim
        
        self.num_layer = num_layer
        
        self.dropout = dropout
        
        self.embedding = embedding
        
        if type(self.embedding)==type(None):
                
            self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=embedding_dim)
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layer,
                            dropout=dropout, bidirectional=True)
        
    def forward(self, input_seq, input_lengths, h=None):
        
        # input_seq (seq, batch)
        # input_lengths (batch)
        # embedded (seq, batch, embedding_dim)
        # outputs (seq, batch, 2*hidden_dim)
        # h (1, batch, 2* hidden_size)
        
        input_seq = input_seq.type(torch.long)
        
        if type(self.embedding)==type(None):
            
            embedded = self.embedding(input_seq)
            
        else:

            seq, batch_size = input_seq.size()
   
            embedded = input_seq.unsqueeze(-1).repeat(1, 1, self.embedding_dim).type(torch.float32)
            
            for i in range(seq):
                
                embedded[i,:,:] = self.embedding[input_seq[i]]
                
            
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)

        if h==None: 
            outputs, h= self.gru(packed)
        else:
            outputs, h = self.gru(packed, h)
        
        #outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        #outputs = outputs[:,:,:self.hidden_dim] + outputs[:,:,self.hidden_dim:]
        
        h = h.view(self.num_layer, 2, batch_size, self.hidden_dim).contiguous()[-1:,:,:,:].transpose(0, 1).view(1, batch_size, 2*self.hidden_dim).contiguous()
            
        return h