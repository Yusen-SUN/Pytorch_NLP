import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layer=2, dropout=0.1, pretrained_embedding=False):
        super(EncoderRNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        
        self.hidden_dim = hidden_dim
        
        self.num_layer = num_layer
        
        self.dropout = dropout
        
        self.pretrained_embedding = pretrained_embedding
        
        if not self.pretrained_embedding:
                
            self.embedding = nn.Embedding(num_embeddings=input_vocab_size, embedding_dim=embedding_dim)
        
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layer,
                            dropout=dropout, bidirectional=True)
        
    def forward(self, input_seq, input_lengths, embedding, h=None, c=None):
        
        # input_seq (seq, batch)
        # input_lengths (batch)
        # embedded (seq, batch, embedding_dim)
        # outputs (seq, batch, hidden_dim)
        # h (num_layers, batch,  hidden_size)
        
        input_seq = input_seq.type(torch.long)
        
        if not self.pretrained_embedding:
            
            embedded = self.embedding(input_seq)
            
        else:

            seq, batch_size = input_seq.size()
   
            embedded = input_seq.unsqueeze(-1).repeat(1, 1, self.embedding_dim).type(torch.float32)
            
            for i in range(seq):
                
                embedded[i,:,:] = embedding[input_seq[i]]
            
            #embedded = embedded.type(torch.float32).to(device)
            
            
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu().numpy())

        if h==None: 
            outputs, (h, c) = self.lstm(packed)
        else:
            outputs, (h, c) = self.lstm(packed, (h, c))
        
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        outputs = outputs[:,:,:self.hidden_dim] + outputs[:,:,self.hidden_dim:]
        
        h = h.view(self.num_layer, 2, -1, self.hidden_dim).contiguous()
    
        c = c.view(self.num_layer, 2, -1, self.hidden_dim).contiguous()
        
        h = h[:,0,:,:] + h[:,1,:,:]
        c = c[:,0,:,:] + c[:,1,:,:]
        
        return outputs, h, c