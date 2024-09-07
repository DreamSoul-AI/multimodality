import torch
import torch.nn as nn
from torch.nn import functional as F
class Sequence(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_size,num_layers,batch_size):
        super(Sequence, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size

    def forward(self, x, hidden=None):
        if hidden == None:
            hidden = self.init_hidden(x.shape[0])
        x = self.embed(x)

        out, (h_n, c_n) = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.linear(out)
        return out
    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        return h0, c0