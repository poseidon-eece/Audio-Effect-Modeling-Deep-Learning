import os
import os.path
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from wavenet_modules import *
from paired_audio_data import *
from audio_utils import quantize_data, mu_law_expansion


class WaveNetModel(nn.Module):
   
    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=512,
                 end_channels=256,
                 classes=256,
                 output_length=16,
                 mu_law_classes=256,
                 kernel_size=2,
                 bias=False,
                 device=torch.device("cpu")):

        super(WaveNetModel, self).__init__()
        self.mu_law_classes = mu_law_classes
        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.classes = mu_law_classes
        self.kernel_size = kernel_size
        self.device = device

        self.dilation_list = []

        self.filter_convs    = nn.ModuleList()
        self.gate_convs      = nn.ModuleList()
        self.residual_convs  = nn.ModuleList()
        self.skip_convs      = nn.ModuleList()
        self.filter_norms = nn.ModuleList()
        self.gate_norms   = nn.ModuleList()

        # 1x1 convolution
        self.start_conv = nn.Conv1d(
            in_channels=self.classes,
            out_channels=residual_channels,
            kernel_size=1,
            bias=bias
        )

        receptive_field = 1

        for b in range(blocks):
            new_dilation = 1
            for i in range(layers):
                self.dilation_list.append(new_dilation)

                padding = (kernel_size - 1) * new_dilation

                self.filter_convs.append(nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=kernel_size,
                    dilation=new_dilation,
                    padding=padding,
                    bias=bias
                ))

                self.gate_convs.append(nn.Conv1d(
                    in_channels=residual_channels,
                    out_channels=dilation_channels,
                    kernel_size=kernel_size,
                    dilation=new_dilation,
                    padding=padding,
                    bias=bias
                ))

                self.filter_norms.append(nn.LayerNorm(dilation_channels))
                self.gate_norms.append(nn.LayerNorm(dilation_channels))

                # 1x1 residual connection
                self.residual_convs.append(nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=residual_channels,
                    kernel_size=1,
                    bias=bias
                ))

                # 1x1 skip connection
                self.skip_convs.append(nn.Conv1d(
                    in_channels=dilation_channels,
                    out_channels=skip_channels,
                    kernel_size=1,
                    bias=bias
                ))

                receptive_field += (kernel_size - 1) * new_dilation
                new_dilation *= 2

        self.end_conv_1 = nn.Conv1d(
            in_channels=skip_channels,
            out_channels=end_channels,
            kernel_size=1,
            bias=True
        )

        self.end_conv_2 = nn.Conv1d(
            in_channels=end_channels,
            out_channels=classes,
            kernel_size=1,
            bias=True
        )

        self.output_length = output_length
        self.receptive_field = receptive_field

        #Kaiming Weight Initialization
        self._initialize_weights()

        self.to(self.device)

    def _initialize_weights(self):
        
        # start_conv: linear → relu context
        nn.init.kaiming_normal_(self.start_conv.weight, nonlinearity='relu')

        for i in range(len(self.filter_convs)):
            # filter/gate convs
            nn.init.kaiming_normal_(self.filter_convs[i].weight, nonlinearity='tanh')
            nn.init.kaiming_normal_(self.gate_convs[i].weight,   nonlinearity='tanh')

            # residual/skip
            nn.init.kaiming_normal_(self.residual_convs[i].weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.skip_convs[i].weight,     nonlinearity='relu')

        # end convs
        nn.init.kaiming_normal_(self.end_conv_1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.end_conv_2.weight, nonlinearity='relu')
        nn.init.zeros_(self.end_conv_1.bias)
        nn.init.zeros_(self.end_conv_2.bias)

    def wavenet(self, input):
    
        input = input.float()
        x = self.start_conv(input)   # [B, residual_channels, T]
        skip = 0

        for i in range(self.blocks * self.layers):

            residual = x

            # Dilated convolutions
            filter_out = self.filter_convs[i](x)
            gate_out   = self.gate_convs[i](x)

            # Κόβουμε τα extra timesteps από το causal padding
            T = residual.size(2)
            filter_out = filter_out[:, :, :T]   # [B, dilation_channels, T]
            gate_out   = gate_out[:, :, :T]     # [B, dilation_channels, T]

            filter_out = self.filter_norms[i](
                filter_out.transpose(1, 2)   # [B, T, dilation_channels]
            ).transpose(1, 2)                # πίσω σε [B, dilation_channels, T]

            gate_out = self.gate_norms[i](
                gate_out.transpose(1, 2)
            ).transpose(1, 2)

            # Gated activation
            x = torch.tanh(filter_out) * torch.sigmoid(gate_out)

            # Skip connection
            s = self.skip_convs[i](x)
            skip = s if i == 0 else skip[:, :, :s.size(2)] + s

            # Residual connection
            x = self.residual_convs[i](x) + residual

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x

    def forward(self, clean_input):
        """
        clean_input: [B, 256, T]
        output:      [B * output_length, 256]
        """
        x = self.wavenet(clean_input)
        x = x[:, :, -self.output_length:]      # [B, 256, output_length]
        x = x.transpose(1, 2).contiguous()     # [B, output_length, 256]
        x = x.view(-1, self.classes)           # [B*output_length, 256]
        return x

    def generate(self, clean_audio_path: str, sampling_rate: int, device, temperature=0.5):
    self.eval()
    
    clean_y, _ = lr.load(clean_audio_path, sr=sampling_rate, mono=True)
    
    q_clean = quantize_data(clean_y, self.classes)
    clean_tensor = torch.from_numpy(q_clean).float().to(device)
    clean_one_hot = torch.zeros(self.classes, clean_tensor.size(0), device=device)
    clean_one_hot.scatter_(0, clean_tensor.unsqueeze(0).long(), 1.)
    clean_one_hot = clean_one_hot.unsqueeze(0)  # [1, C, L]
    
    output_segments = []
    chunk_size = self.receptive_field + self.output_length
    
    with torch.no_grad():
        for start in range(0, clean_one_hot.size(2) - chunk_size, self.output_length):
            segment = clean_one_hot[:, :, start:start + chunk_size]
            y_pred = self.forward(segment)
            y_pred = y_pred.reshape(-1, self.classes)
            probabilities = F.softmax(y_pred / temperature, dim=1)
            predicted = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            output_segments.append(predicted.cpu().numpy())
    
    predicted_indices = np.concatenate(output_segments)
    decoded_audio = mu_law_expansion(predicted_indices)
    
    return decoded_audio, sampling_rate