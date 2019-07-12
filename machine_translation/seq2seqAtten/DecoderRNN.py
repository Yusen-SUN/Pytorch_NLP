import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class DecoderRNN(nn.Module):
       
    def __init__(self, attn_model, output_vocab_size, embedding_dim, hidden_dim, num_layer, dropout, pretrained_embedding=False):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.embedding_dim = embedding_dim
        
        self.num_layer = num_layer
        
        self.output_vocab_size = output_vocab_size
        
        self.pretrained_embedding = pretrained_embedding
        
        if not pretrained_embedding:
        
            self.embedding = nn.Embedding(output_vocab_size, embedding_dim)   
        
        self.embedding_dropout = nn.Dropout(dropout) 
        
        self.lstm = nn.LSTM(input_size=(hidden_dim+embedding_dim), hidden_size=hidden_dim, num_layers=num_layer, dropout=dropout)
                
        self.out = nn.Linear(hidden_dim, output_vocab_size)
        
        self.attn = attn_model
        
        
    def forward_word(self, input_word, embedding, h, c, encoder_outputs):
        
        # input_word (1, batch)
        # h (num_layers, batch, hidden_size)
        # c (num_layers, batch, hidden_size)    
        # encoder_outputs (seq, batch, hidden_dim)
        
                
        # output (batch, output_vocab_size)
        # attn_weights (batch, 1, seq)
        
        input_word = input_word.type(torch.long)
        
        if not self.pretrained_embedding:
        
            # embedded (1, batch, embedd_dim)
            embedded = self.embedding(input_word)
            
        else:
            # embedded (1, batch, embedd_dim)
            embedded = embedding[input_word[0,:]].unsqueeze(0)#.type(torch.float32).to(device)
        
        # embedded (1, batch, embedd_dim)
        embedded = self.embedding_dropout(embedded)
        
        # h_atten (1, batch, hidden_dim)
        h_atten = h[-1,:,:].unsqueeze(0)
        
         # attn_weights  (batch, 1, seq)
        attn_weights = self.attn.forward(h_atten, encoder_outputs)
        
        # encoder_outputs (batch, seq, hidden_dim)
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)

        # conect (1, b, hidden)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1).unsqueeze(0)
        
        # concat_input (batch, hidden_dim+embedding_dim)
        concat_input = torch.cat((embedded, context), dim=-1)
        
        # concat_output (batch, hidden_dim)
        #concat_output = torch.relu(self.concat(concat_input))
          
        # output (1, batch, hidden_dim)
        # h (2, batch, hidden_dim)
        # c (2, batch, hidden_dim)
        output, (h, c) = self.lstm(concat_input, (h, c))    
        
        # output (batch, hidden_dim)
        output = output.squeeze(0)
        
        # output (batch, output_vocab_size)
        output = self.out(output)
        
        # output (batch, output_vocab_size)
        output = F.log_softmax(output, dim=-1)
        
        # output (batch, output_vocab_size)
        # h (2, batch, hidden_dim)
        # c (2, batch, hidden_dim)
        # attn_weights  (batch, atten_seq)
        
        return output, h, c, attn_weights.squeeze(1)
    
    def forward(self, embedding, h, c, encoder_outputs):
        
        target_length, batch_size, _ = encoder_outputs.size()
        
        decoder_input = torch.ones(batch_size)
        
        outputs = []
        
        output_seq = []
        
        output_atten = []

        for i in range(0, target_length):

            decoder_input = decoder_input.unsqueeze(0)

            decoder_output, h, c, attn_weights = self.forward_word(decoder_input, embedding, h, c, encoder_outputs)

            topv, topi = decoder_output.topk(1)

            decoder_input = topi.detach().squeeze()
            
            outputs.append(decoder_output.unsqueeze(1))
            
            output_seq.append(decoder_input.unsqueeze(1))

            output_atten.append(attn_weights.detach().unsqueeze(1))
        
        
        output = torch.cat(outputs, dim=1)
        seq = torch.cat(output_seq, dim=1)
        attn = torch.cat(output_atten, dim=1)
        
        return output, seq, attn
    
    
    
