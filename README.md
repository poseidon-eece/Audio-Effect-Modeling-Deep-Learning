# Audio Effect Modeling with Deep Learning

This repository contains the implementation of my undergraduate thesis at the **Democritus University of Thrace (DUTH)**. The project focuses on simulating guitar overdrive effects using three distinct Deep Learning architectures.
Note: This project is currently a Work in Progress (WIP), full thesis document upload soon

## Overview
The goal is to map "clean" guitar signals to "processed" (overdrive) signals using the [EGFX Dataset](https://egfxset.github.io/).

## Architectures & Loss Functions
We implemented and compared three different models:

| Model | Approach | Loss Function |
| :--- | :--- | :--- |
| **WaveNet** | Autoregressive / Mu-law Classification | Cross-Entropy Loss |
| **GRU** | Sequence-to-Sequence Regression | Multi-Resolution STFT Loss |
| **LSTM** | Sequence-to-Sequence Regression | Combined ESR (Error-to-Signal) & DC Loss |

## Project Structure
- `/models`: Source code for each architecture.
- `/notebooks`: Google Colab demos for training and inference.

## Key Features 
- **Mu-law Companding & 8-bit Quantization:** Used in WaveNet for transforming continuous audio into a discrete classification problem.
- **Custom Paired Audio Dataset Pipeline:** Automatic creation of aligned clean/processed `.npz` datasets with sample-level synchronization and efficient memory mapping.
- **One-Hot Conditioning for Audio Input:** Clean signal encoded as one-hot vectors to condition the WaveNet model.
- **Dilated Causal Convolutions:** Exponentially increasing receptive field enabling long-range temporal dependencies in audio modeling.
- **Gated Activation Units (Tanh × Sigmoid):** Core mechanism for nonlinear audio transformation in WaveNet.
- **Residual & Skip Connections:** Stable deep training and efficient gradient flow across very deep architectures.
- **Multi-Resolution STFT Loss (MRSTFT):** Perceptually meaningful loss for audio quality optimization (used in GRU model).
- **Custom ESR + DC Loss Combination:** Tailored loss for waveform fidelity and DC offset correction in LSTM model.
- **Mixed Precision Training (AMP):** Faster training and reduced GPU memory usage using automatic mixed precision.
- **Gradient Clipping:** Stabilization of training for recurrent and autoregressive models.
- **Learning Rate Scheduling (ReduceLROnPlateau):** Adaptive learning rate tuning based on validation loss.
- **Audio Generation Pipeline:** End-to-end inference including sampling, temperature control, and inverse mu-law decoding.
- **TensorBoard Logging:** Real-time monitoring of loss, gradients, and model behavior.
- **Modular Training Framework:** Reusable trainer, logger, and model structure across all architectures.
