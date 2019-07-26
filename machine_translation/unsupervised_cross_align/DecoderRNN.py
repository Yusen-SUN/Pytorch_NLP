import torch
import torch.nn as nn
import torch.nn.functional as F
from Attention import Attn

class DecoderRNN(nn.Module):
       
    def __init__(self, attn_model, vocab_size, embedding_dim, hidden_dim, device, num_layer, dropout):#, embedding=None):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        
        self.embedding_dim = embedding_dim
        
        self.num_layer = num_layer
        
        self.output_vocab_size = vocab_size
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        self.device = device
                
       #self.embedding = embedding if type(embedding)==type(None) else nn.Embedding(num_embeddings=output_vocab_size, embedding_dim=embedding_dim)
        
        self.embedding_dropout = nn.Dropout(dropout) 
        
        self.gru = nn.GRU(input_size=(self.hidden_dim+embedding_dim), hidden_size=self.hidden_dim, num_layers=self.num_layer, dropout=dropout)
                
        self.out = nn.Linear(self.hidden_dim, vocab_size)
        
        self.attn = Attn('general', self.hidden_dim)
        
    def forward_word(self, input_word, h, encoder_outputs):
        
        # input_word (1, batch)
        # h (num_layer, batch, hidden_size)
        # encoder_outputs (seq, batch, hidden_dim)

        # output (batch, output_vocab_size)
        # attn_weights (batch, 1, seq)
        
        input_word = input_word.type(torch.long)
        
        #embedded (1, batch, embedd_dim)
        embedded = self.embedding(input_word)
        
#         if type(self.embedding)==type(None):
        
#             # embedded (1, batch, embedd_dim)
#             embedded = self.embedding(input_word)
            
#         else:
#             # embedded (1, batch, embedd_dim)
#             embedded = self.embedding[input_word[0,:]].unsqueeze(0)
        
        # embedded (1, batch, embedd_dim)
        embedded = self.embedding_dropout(embedded)        
        
        # attn_weights  (batch, 1, seq)
        attn_weights = self.attn.forward(h[-1].unsqueeze(0), encoder_outputs)
        
        # encoder_outputs (batch, seq, encoder_hidden_dim)
        encoder_outputs = torch.transpose(encoder_outputs, 0, 1)

        # context (1, b, encoder_hidden_dim)
        context = torch.bmm(attn_weights, encoder_outputs).squeeze(1).unsqueeze(0)
        
        # concat_input (1, batch, hidden_dim+embedding_dim)
        concat_input = torch.cat((embedded, context), dim=-1)
        
        # concat_output (1, batch, hidden_dim+embedding_dim)
        concat_input = F.relu(concat_input)
      
        # output (1, batch, hidden_dim)
        # h (num_layer, batch, hidden_dim)
        output, h, = self.gru(concat_input, h)    
        
        # output (batch, hidden_dim)
        output = output.squeeze(0)
        
        # output (batch, output_vocab_size)
        output = self.out(output)
        
        # output (batch, output_vocab_size)
        # h (num_layer, batch, hidden_dim)
        return output, h
    
    def forward(self, h, encoder_outputs, style):
        
        #h: num_layer, b, hidden
        #encoder_outputs: seq, b, hidden

        target_length, batch_size, _ = encoder_outputs.size()
        
        # decoder_input: b
        decoder_input = torch.ones(batch_size).to(self.device)*style
        
        outputs = []
        output_seq = []
        #output_embedding = []
            
        for i in range(0, target_length-1):

            # decoder_input: 1, b
            decoder_input = decoder_input.unsqueeze(0)

            # decoder_output: b, output_vocab_size
            # h: num_layer, batch, hidden
            decoder_output, h = self.forward_word(decoder_input, h, encoder_outputs)
            
            # output (batch, output_vocab_size)
            #decoder_output_prob = F.softmax(decoder_output, dim=-1)
            # output (1, batch, embeded_dim)
            #decoder_embedding =  torch.bmm(decoder_output_prob, self.embedding).unsqueeze(0)
            
            # decoder_output: b, output_vocab_size
            decoder_output = F.log_softmax(decoder_output, dim=-1)
            
            # topi, topv: b, 1
            topv, topi = decoder_output.topk(1)
            
            #decoder_input: b
            decoder_input = topi.detach().squeeze()
            
            outputs.append(decoder_output.unsqueeze(1))
            output_seq.append(decoder_input.unsqueeze(1))
            #output_embedding.append(decoder_embedding.squeeze(0).unsqueeze(1))
        
        # outputs: batch, target_length, output_vocab_size
        output = torch.cat(outputs, dim=1)
        
        # seq: b, target_length
        seq = torch.cat(output_seq, dim=1)
        
        # output_embeddings: batch, target_length, embedd_dim
        #output_embeddings = torch.cat(output_seq, dim=1)
        
        return output, seq#, output_embeddings
    
    
    
