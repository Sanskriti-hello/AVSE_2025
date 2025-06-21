import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math

class SqueezeformerAVSE(nn.Module):
    """Squeezeformer AVSE Backbone (Supports Stable Diffusion or Linear Decoder)"""
    def __init__(self, sample_rate=16000):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512

        # Audio encoder
        self.audio_patch_embed = nn.Conv1d(257, 512, kernel_size=4, stride=4)
        self.audio_pos_encoding = PositionalEncoding(512)
        self.audio_blocks = nn.ModuleList([
            SqueezeformerBlock(512, 8, downsample=(i == 3)) for i in range(8)
        ])

        # Video encoder
        self.video_backbone = nn.Sequential(
            nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, (1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 7, 7))
        )
        self.video_transformer = nn.Sequential(
            *[TransformerBlock(128 * 7 * 7, 8) for _ in range(4)],
            nn.Linear(128 * 7 * 7, 512)
        )

        # Cross-modal fusion
        self.cross_attn = BidirectionalCrossAttention(512, 8, 4)

        # Choose one: diffusion decoder or linear decoder
        self.diffusion_decoder = None
        self.linear_decoder = LinearDecoder(512, 257)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv3d)):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def set_diffusion_decoder(self, decoder):
        self.diffusion_decoder = decoder

    def forward(self, mixture_audio, target_video):
        # Audio
        mixture_spec = self.audio_to_spec(mixture_audio)
        audio_patches = self.audio_patch_embed(mixture_spec)
        audio_features = audio_patches.transpose(1, 2)
        audio_features = self.audio_pos_encoding(audio_features)

        for block in self.audio_blocks:
            audio_features = block(audio_features)

        # Video
        B, T_v, C, H, W = target_video.shape
        video_features = self.video_backbone(target_video.transpose(1, 2))
        video_features = video_features.permute(0, 2, 1, 3, 4).reshape(B, T_v, -1)
        video_features = self.video_transformer(video_features)

        # Fusion
        fused_features = self.cross_attn(audio_features, video_features)

        # Decode
        if self.diffusion_decoder is not None:
            enhanced_spec = self.diffusion_decoder(fused_features)
        else:
            enhanced_spec = self.linear_decoder(fused_features, mixture_spec)

        return self.spec_to_audio(enhanced_spec)

    def audio_to_spec(self, audio):
        spec = torch.stft(audio, self.n_fft, self.hop_length, self.win_length,
                          window=torch.hann_window(self.win_length).to(audio.device), return_complex=True)
        return torch.abs(spec)

    def spec_to_audio(self, spec_magnitude):
        phase = torch.angle(torch.randn_like(spec_magnitude, dtype=torch.complex64))
        complex_spec = spec_magnitude * torch.exp(1j * phase)
        return torch.istft(complex_spec, self.n_fft, self.hop_length, self.win_length,
                           window=torch.hann_window(self.win_length).to(spec_magnitude.device))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class SqueezeformerBlock(nn.Module):
    def __init__(self, dim, heads, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.conv = nn.Conv1d(dim, dim, 31, padding=15, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(),
            nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )
        if downsample:
            self.downsample_layer = nn.Conv1d(dim, dim, 2, stride=2)

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + x_conv
        x = x + self.ffn(self.norm2(x))
        if self.downsample:
            x = self.downsample_layer(x.transpose(1, 2)).transpose(1, 2)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim), nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim, heads, layers):
        super().__init__()
        self.layers = layers
        self.audio_to_video = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)])
        self.video_to_audio = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(layers * 2)])

    def forward(self, audio, video):
        if audio.size(1) != video.size(1):
            video = F.interpolate(video.transpose(1, 2), size=audio.size(1), mode='linear', align_corners=False).transpose(1, 2)
        for i in range(self.layers):
            a2v, _ = self.audio_to_video[i](audio, video, video)
            v2a, _ = self.video_to_audio[i](video, audio, audio)
            audio = audio + self.norms[i * 2](v2a)
            video = video + self.norms[i * 2 + 1](a2v)
        return (audio + video) / 2

class LinearDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, features, mixture_spec):
        mask = torch.sigmoid(self.layers(features)).transpose(1, 2)  # [B, F, T]
        return mixture_spec * mask
