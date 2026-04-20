import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

def complex_to_magnitude(stft_output):
    """Calculates the magnitude of the complex STFT output."""
    # stft_output shape: (B, F, T, 2) where 2 is (real, imag)
    return torch.sqrt(stft_output[..., 0]**2 + stft_output[..., 1]**2)

class MRSTFTLoss(nn.Module):
    """
    Multi-Resolution Short-Time Fourier Transform (MRSTFT) Loss.
    Calculates the L1 loss on the magnitude spectrum across multiple STFT resolutions.
    """
    def __init__(self,
                 scales: List[int] = [8192, 2048, 512, 128],
                 overlap: float = 0.75,
                 eps: float = 1e-7):
        super().__init__()
        self.scales = scales
        self.overlap = overlap
        self.eps = eps
        self.num_scales = len(scales)
        
        # Create a list of windows for each scale
        self.windows = nn.ParameterList()
        for scale in self.scales:
            # Use Hanning window as is standard in audio processing
            window = torch.from_numpy(np.hanning(scale)).float()
            self.windows.append(nn.Parameter(window, requires_grad=False))

    def forward(self, predict: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculates the MRSTFT Loss.
        predict, target: Tensors of shape (B, T) or (B, 1, T)
        """
        # Ensure input is 2D (B, T)
        if predict.dim() == 3:
            predict = predict.squeeze(-1)
        if target.dim() == 3:
            target = target.squeeze(-1)


        total_loss = 0.0
        
        for i in range(self.num_scales):
            win_len = self.scales[i]
            window = self.windows[i].to(predict.device)
            hop_length = int((1 - self.overlap) * win_len)
            n_fft = win_len # n_fft is typically equal to window length

            # 1. Calculate STFT for prediction and target
            stft_predict = torch.stft(
                predict,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_len,
                window=window,
                return_complex=False, # Return (real, imag) tuple
                center=True # Pad input for centered STFT
            )
            
            stft_target = torch.stft(
                target,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_len,
                window=window,
                return_complex=False,
                center=True
            )

            # 2. Convert to Magnitude Spectrum
            stft_predict_mag = complex_to_magnitude(stft_predict)
            stft_target_mag = complex_to_magnitude(stft_target)

            # 3. Calculate L1 Loss on Magnitude Spectrum
            mag_loss = F.l1_loss(stft_predict_mag, stft_target_mag)
            
            # 4. Add to total loss
            total_loss += mag_loss

        return total_loss / self.num_scales
