import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa as lr

class GRUModel(nn.Module):
    """
    GRU-based model for audio effect modeling (sequence-to-sequence regression).
    Takes float audio input [-1, 1] and outputs float audio [-1, 1].
    """
    def __init__(self,
                 input_size=1,          # 1 for mono audio sample
                 hidden_size=128,       # Size of the hidden state
                 num_layers=2,          # Number of stacked GRU layers
                 output_size=1,         # 1 for mono audio sample output
                 dropout=0.2,
                 bidirectional=False,
                 device=torch.device("cpu")):
        
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.num_directions = 2 if bidirectional else 1
        
        # GRU Layer
        # batch_first=True: Input/Output shape is (Batch, Sequence Length, Features)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout, 
                          bidirectional=bidirectional)
        
        # Maps the hidden state (hidden_size * num_directions) to the output sample (1)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)
        
        # Activation function for output (tanh to keep output in [-1, 1] range)
        self.output_activation = nn.Tanh()
        
        self.to(self.device)

    def forward(self, input_seq):
        # input_seq shape: (B, L, 1) - Batch, Length, Features (1)
        
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, 
                         input_seq.size(0), 
                         self.hidden_size).to(self.device)
        
        # Forward propagate GRU
        # output shape: (B, L, hidden_size * num_directions)
        # hn shape: (num_layers * num_directions, B, hidden_size)
        output, hn = self.gru(input_seq, h0)
        
        # output shape: (B * L, hidden_size * num_directions)
        output = output.contiguous().view(-1, output.size(2))
        
        # output shape: (B * L, 1)
        output = self.fc(output)
        
        # input_seq shape: (B, L, 1) -> view(-1, 1) -> (B*L, 1)
        input_seq_reshaped = input_seq.contiguous().view(-1, 1)
        
        # Final output = Input + Residual (GRU output before Tanh)
        final_output = input_seq_reshaped + output
        
        # Apply Tanh activation to the final output to ensure it stays in [-1, 1]
        final_output = self.output_activation(final_output)
        
        return final_output.view(input_seq.size(0), input_seq.size(1), -1) # Returns [B, L, 1]

    def generate(self, clean_audio_path: str, sampling_rate: int, device):
        """
        Generates the processed audio signal from a clean audio file.
        Uses the full sequence forward pass (non-autoregressive).
        """
        self.eval()
        
        # 1. Load Clean Audio (float, [-1, 1])
        clean_y, sr = lr.load(clean_audio_path, sr=sampling_rate, mono=True)
        
        # Convert numpy array to tensor
        clean_tensor = torch.from_numpy(clean_y).float().to(device)
        
        # Reshape to (1, L, 1) -> (Batch=1, Length, Features=1)
        input_seq = clean_tensor.unsqueeze(0).unsqueeze(-1)
        
        # 3. Forward Pass
        with torch.no_grad():
            # output shape: (L, 1) - Reshaped from (1*L, 1)
            output = self.forward(input_seq).squeeze(-1) 
            
        # 4. Convert to numpy and return
        decoded_audio = output.cpu().numpy()
        
        return decoded_audio, sampling_rate