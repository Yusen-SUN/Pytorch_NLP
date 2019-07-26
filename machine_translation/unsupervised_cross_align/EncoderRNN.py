import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, device, num_layer=2, dropout=0.5, bidirectional=True):
        #, same_embedding=True, pretrained_embedding=None):
        
        super(EncoderRNN, self).__init__()
        
        if bidirectional and hidden_dim%2!=0 :
             raise ValueError('The hidden dimension needs to be even for bidirectional encoders !')
        
        self.vocab_size = vocab_size
        
        #self.vocab_size_2 = vocab_size_2
        
        self.bidirectional = bidirectional
        
        self.embedding_dim = embedding_dim
        
        self.direction = 2 if bidirectional else 1
        
        # hidden dim for the first layer
        self.hidden_dim_1 = hidden_dim//self.direction 
        
        self.hidden_dim = hidden_dim
                
        self.num_layer = num_layer
        
        self.dropout = dropout
        
        self.device = device

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        self.dropout = nn.Dropout(dropout)
        
        # gru for the first layer
        self.gru_1 = nn.GRU(input_size=embedding_dim, hidden_size=self.hidden_dim_1, num_layers=1, bidirectional=bidirectional)
        
        # gru for the rest layers
        if num_layer > 1:
            self.gru = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim, num_layers=num_layer-1, dropout=dropout, bidirectional=False)
        
#         if type(pretrained_embedding)==type(None):
#             if same_embedding:
#                 self.embedding = nn.Embedding(num_embeddings=vocab_size_1, embedding_dim=embedding_dim, padding_idx=0)
#             else:
#                 self.embedding_1 = nn.Embedding(num_embeddings=vocab_size_1, embedding_dim=embedding_dim, padding_idx=0)
#                 self.embedding_2 = nn.Embedding(num_embeddings=vocab_size_2, embedding_dim=embedding_dim, padding_idx=0)
            
# #             self.embeddingBag_1 = nn.EmbeddingBag.(vocab_size_1, embedding_dim, mode='sum')
# #             self.embeddingBag_2 = nn.EmbeddingBag.(vocab_size_2, embedding_dim, mode='sum')
#         else:
#             if same_embedding:
#                 self.embedding = nn.Embedding.from_pretrained(pretrained_embedding.embedding, freeze=True, padding_idx=0)
#             else:
#                 self.embedding_1 = nn.Embedding.from_pretrained(pretrained_embedding[0].embedding, freeze=True, padding_idx=0)
#                 self.embedding_2 = nn.Embedding.from_pretrained(pretrained_embedding[1].embedding, freeze=True, padding_idx=0)
#                 self.embeddingBag_1 = nn.EmbeddingBag.from_pretrained(pretrained_embedding[0].embedding, freeze=True, mode='sum')
#                 self.embeddingBag_2 = nn.EmbeddingBag.from_pretrained(pretrained_embedding[1].embedding, freeze=True, mode='sum')
        

        
    def processing_input(self, input_seq, input_lengths, style):
        # input_seq: batch, seq
        # input_lengths: batch
        # style: 1
        
        # input_seq: seq, batch
        input_seq = input_seq.t()
        
        # token for style
        # style_tensor: 1, batch
        style_tensor = torch.ones(1, input_seq.size()[1]).type(torch.long).to(self.device)*style
        input_lengths = input_lengths + 1
        
        # new_input_seq: seq+1, batch
        new_input_seq = torch.cat([style_tensor, input_seq], dim=0)
        
        #embedded: seq+1, batch, embedding_dim
        embedded = self.embedding(new_input_seq)
        
#         # input_seq: seq, batch
#         if len(input_seq.size())==2:
#             # new_input_seq: seq+1, batch
#             new_input_seq = torch.cat([style_tensor, input_seq], dim=0)
#             # embedded: seq+1, batch, embedding_dim
#             #embedded = self.embedding_1(new_input_seq) if style==1 else self.embedding_1(new_input_seq)
            

#         # input_seq: seq, batch, vocab_size       
#         elif len(input_seq.size())==3:
            
#              # input_seq: batch, seq, vocab_size
#             input_seq = torch.transpose(input_seq, 0, 1)
            
#             vocab_size = self.vocab_size_1 if style==1 else self.vocab_size_2
            
#             # 1, batch, embedd_dim
#             style_embedded = self.embedding[style_tensor[0].unsqueeze(0)]
            
#             input = torch.ones(sizes=[seq_len*batch_size, vocab_size], , dtype=torch.long)
#             offsets = torch.LongTensor(range(0, seq_len*batch_size, seq_len))
#             per_sample_weights = torch.flatten(input_seq)
            
#             # batch_size, seq_len, embedding_dim
#             embedded = self.embeddingBag_1(input, offsets, per_sample_weights) if style==1 else self.embeddingBag_2(input, offsets, per_sample_weights)
            
#             embedded = embedded.view(batch_size, seq_len, self.embedding_dim).contiguous()
            
#             embedded = torch.cat([style_embedded, torch.transpose(embedded, 0 ,1)])
            
#         else:
#             raise ValueError('Incorrect input dim for encoder!')
                    
        return embedded, input_lengths
    
    def forward(self, input_seq, input_lengths, style):
        # input_seq (batch, seq)
        # input_lengths (batch)
        # style: 1

        batch_size = input_seq.size()[0]
        
        # embedded (seq+1, batch, embedding_dim)
        # input_lengths (batch)
        embedded, input_lengths = self.processing_input(input_seq, input_lengths, style)
         
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        
        #h_1: (1 * num_directions, batch, hidden_dim_1)
        outputs, h_1 = self.gru_1(packed)
        
        if self.bidirectional:
            #h: (1, batch, hidden_size)
            h_1 = torch.cat([h_1[0,:,:], h_1[1,:,:]], dim=-1).unsqueeze(0)
        
        if self.num_layer > 1:

            outputs, lens = nn.utils.rnn.pad_packed_sequence(outputs)
            outputs = self.dropout(outputs)
            packed = nn.utils.rnn.pack_padded_sequence(outputs, lens)
            
            # h_rest: (num_layers-1, batch, hidden_dim)
            outputs, h_rest = self.gru(packed)
            
            # h: (num_layers, batch, hidden_dim)
            h = torch.cat([h_1, h_rest], dim=0)
        else:
            h = h_1
        
        #outputs (seq, batch, hidden_size)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        
        #outputs (seq, batch, hidden_size)
        #h: (num_layers, batch, hidden_size)
        return outputs, h