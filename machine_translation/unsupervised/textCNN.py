import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, seq_length, embedding_size, num_labels, embedding=None, filter_sizes=[3, 4, 5], drop_out_rate=0.5, num_feature_maps=500):
        
        super(textCNN, self).__init__()
        
        self.embedding_dim = embedding_size

        self.embedding = embedding
 
        self.dropout = nn.Dropout(drop_out_rate)
    
        self.conv = nn.ModuleList([ 
            nn.Sequential(
                # input size: batch, in_channels, seq_length, embedding_size
                # output size: batch, out_channels, seq_length-filter_size+1, 1
                nn.Conv2d(in_channels=1, out_channels=num_feature_maps, kernel_size=(i, embedding_size)),
                nn.ReLU(),
                # input size: batch, out_channels, seq_length-filter_size+1, 1
                # output size: batch, out_channels, 1, 1
                nn.MaxPool2d(kernel_size=(seq_length-i+1, 1))
                ) 
            for i in filter_sizes])
        
        self.out = nn.Linear(num_feature_maps*len(filter_sizes), num_labels)
        
    def processed_input(self, input_seq):
                
        # inputs is seq of index
#         if len(input_seq.size())==2:
        # input: batch, seq
        input_seq = input_seq.type(torch.long)
#             if type(self.embedding)==type(None):
        embedded = self.embedding(input_seq)
#             else:
#                 embedded = input_seq.unsqueeze(-1).repeat(1, 1, self.embedding_dim).type(torch.float32)
#                 for i in range(len(input_seq)):
#                     embedded[i,:,:] = self.embedding[input_seq[i]]
                    
#         # input is seq of word embedding         
#         elif len(input_seq.size())==3:
#             embedded = input_seq
            
#         else:
#             raise ValueError('Incorrect input dim for encoder!')
        
        # embedded: batch, seq_length, embedding_size
        #embedded = torch.transpose(embedded, 0, 1)
        #
        return embedded

    def forward(self, input_seq):
        
        embedded = self.processed_input(input_seq)
        x = [conv(embedded.unsqueeze(1)).squeeze() for conv in self.conv]
        # x: batch, num_feature_maps & num_filters
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x