'''
speaker_encoder.py
Implements a speaker encoder model 
as described in "Neural Voice Cloning with a Few Samples": https://arxiv.org/pdf/1802.06006
'''

import math
import torch
import torch.nn as nn

class FcEluBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.elu = nn.ELU()
    
    def forward(self, x):
        x = self.fc(x)
        x = self.elu(x)
        return x

class ConvGluBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=12):
        super().__init__()
        self.kernel_size=kernel_size
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=kernel_size, padding=0)
        self.norm = nn.BatchNorm1d(num_features=in_channels*2)
        self.glu = nn.GLU(dim=1)

    def forward(self, x):
        residual = x
        # Pad left and right sides of conv input as described here: https://arxiv.org/pdf/1710.07654
        total_pad = self.kernel_size - 1
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad 
        x = torch.nn.functional.pad(x, (left_pad, right_pad))
        x = self.conv(x)
        x = self.glu(x)
        x = (x + residual) * math.sqrt(0.5)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, mel_dim, hidden_dim, attn_dim, embedding_out_dim, N_prenet=2, N_conv=2):
        super().__init__()
        # FC + ELU x N_prenet
        self.fc_elu_blocks = nn.ModuleList()
        current_dim = mel_dim
        for _ in range(N_prenet):
            self.fc_elu_blocks.append(FcEluBlock(current_dim, hidden_dim))
            current_dim = hidden_dim

        # Conv + GLU + Residual + Scaling × n_conv_glu_blocks
        self.conv_glu_blocks = nn.ModuleList()
        for _ in range(N_conv):
            self.conv_glu_blocks.append(ConvGluBlock(hidden_dim))

        # Temporal Masking
        self.dropout = nn.Dropout(p=0.2)

        # Residual Projection from conv_glue_blocks to output
        self.fc_residual = nn.Linear(hidden_dim, embedding_out_dim)

        # Multi-head Attention Projections
        self.attn_key = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ELU())
        self.attn_query = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ELU())
        self.attn_value = nn.Sequential(nn.Linear(hidden_dim, attn_dim), nn.ELU())

        self.attention = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=2, batch_first=True)

        # Final Projection to Speaker Embedding
        self.fc_output = nn.Linear(attn_dim, embedding_out_dim)
        self.softsign = nn.Softsign()

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, n_speaker_samples, n_audio_samples, n_mel_freqs)
        Returns: speaker embedding (batch_size, embedding_out_dim)
        """
        batch_size, n_speaker_samples, n_audio_samples, n_mel_freqs = x.shape
        x = x.view(batch_size * n_speaker_samples, n_audio_samples, n_mel_freqs)

        for fc_elu_block in self.fc_elu_blocks:
            x = fc_elu_block(x) # resulting shape: (batch_size * n_speaker_samples, n_audio_samples, hidden_dim)

        x = x.transpose(1, 2) # resulting shape: (batch_size * n_speaker_samples, hidden_dim, n_audio_samples)

        for conv_glu_block in self.conv_glu_blocks:
            x = conv_glu_block(x)

        # Temporal Masking
        x = self.dropout(x) 

        # Global Average Pool Over Time Dimension
        x = x.mean(dim=2)   
        x = x.view(batch_size, n_speaker_samples, -1) # resulting shape: (batch_size, n_speaker_samples, hidden_dim)

        # Store output for residual connection (batch_size, embedding_out_dim)
        residual = self.fc_residual(x) 

        # Multi-head attention over n_speaker_samples reference samples
        attn_keys = self.attn_key(x)
        attn_queries = self.attn_query(x)
        attn_values = self.attn_value(x)
        attn_out, _ = self.attention(attn_queries, attn_keys, attn_values)  # resulting shape: (batch_size, n_reference_samples, attn_dim)

        out = self.fc_output(attn_out) # resulting shape: (batch_size, n_reference_samples, embedding_out_dim)
        out = self.softsign(out)
        out = torch.nn.functional.normalize(out, p=2, dim=-1)

        out = out * residual

        return out

