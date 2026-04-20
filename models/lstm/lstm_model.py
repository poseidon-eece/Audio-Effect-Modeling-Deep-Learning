import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, bidirectional=False, dropout=0.2, device='cpu'):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.linear = nn.Linear(fc_input_size, output_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        num_dirs = 2 if self.lstm.bidirectional else 1
        
        h0 = torch.zeros(self.num_layers * num_dirs, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * num_dirs, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x)
        out = self.dropout(out)
        out = self.linear(out)
        
        # Skip Connection (Residual Learning)
        out = out + x
        out = torch.tanh(out)
        return out
