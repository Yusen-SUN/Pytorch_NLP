import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class Attn(nn.Module):
    
    # h (1, b, hidden_dim)
    # encoder_output (seq, b, hidden_dim)
    # attn_energies (b, 1, seq)
    def __init__(self, method, hidden_dim):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_dim = hidden_dim
        
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
            
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_dim, self.hidden_dim)
            
        elif self.method == 'concat':
            self.attn = nn,Linear(self.hidden_dim*2, self.hidden_dim)
            self.v = torch.nn.Parameter(torch.FloatTensor(self.hidden_dim))
     
        
    def dot_score(self, h, encoder_outputs):
        # (seq, b)
        return torch.sum(h * encoder_output, dim=-1)
    
    def general_score(self, h, encoder_outputs):
        # energy (seq, b, hidden_dim)
        energy = self.attn(encoder_outputs)
        # (seq, b)
        return torch.sum(h * energy, dim=-1)
    
    def concat_score(self, h, encoder_outputs):
        # energy (seq, b, hidden_dim)
        energy = self.attn(torch.cat((h.expand(encoder_outputs.size(0), -1, -1), encoder_outputs), dim=2)).tanh()
        # (seq, b)
        return torch.sum(self.v * energy, dim=-1)
        
    def forward(self, h, encoder_outputs):
        
        # h: 1, batch, hidden
        # encoder_outputs, # outputs, seq, batch, hidden_dim
        
        # attn_energies  (seq, b)
        if self.method == 'general':
            attn_energies = self.general_score(h, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(h, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(h, encoder_outputs)
        
        # attn_energies  (b, seq)
        attn_energies = attn_energies.t()
        
        # attn_energies  (b, 1, seq)
        attn_energies = F.softmax(attn_energies, dim=1).unsqueeze(1)

        # attn_energies  (b, 1, seq)
        return attn_energies