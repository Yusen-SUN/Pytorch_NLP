import torch
import torch.nn as nn
import torch.nn.functional as F


class textCNN(nn.Module):
    def __init__(self, vocab_size, seq_length, embedding_size, num_labels, filter_sizes=[3, 4, 5], drop_out_rate=0.5, num_feature_maps=100):
        super(textCNN, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
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

    def forward(self, input_seq):
        
        # input_seq: batch, seq_length
        input_seq = input_seq.type(torch.long)
        # embedded: batch, seq_length, embedding_size
        embedded = self.embedding(input_seq)
        x = [conv(embedded.unsqueeze(1)).squeeze() for conv in self.conv]
        # x: batch, num_feature_maps & num_filters
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.out(x)
        x = F.log_softmax(x, dim=-1)
        return x