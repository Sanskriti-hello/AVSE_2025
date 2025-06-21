import torch
from torch.utils.data import Dataset
import torchaudio
import cv2
import os
import numpy as np
import json
from pathlib import Path

class COGMHEARDataset(Dataset):
    """Dataset loader adapted for your existing COG-MHEAR structure"""
    
    def __init__(self, data_root, split='train', version='avsec4'):
        self.data_root = Path(data_root)
        self.split = split
        self.version = version
        
        # Your existing structure paths
        self.avse_data_path = self.data_root / 'avse_data'
        self.metadata_path = self.avse_data_path / 'metadata'
        
        # Load metadata based on your structure
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load samples from your existing metadata structure"""
        samples = []
        
        # Path to your prepared data
        if self.version == 'avsec4':
            data_prep_path = self.data_root / 'avse_challenge' / 'data_preparation' / 'avsec4'
        else:
            data_prep_path = self.data_root / 'avse_challenge' / 'data_preparation' / 'avsec1'
        
        # Look for your prepared samples
        split_path = data_prep_path / self.split
        if split_path.exists():
            for item in split_path.iterdir():
                if item.is_dir():
                    # Look for audio-video pairs
                    audio_files = list(item.glob('*.wav'))
                    video_files = list(item.glob('*.mp4'))
                    
                    for audio_file in audio_files:
                        # Find corresponding video
                        video_file = audio_file.with_suffix('.mp4')
                        if video_file.exists():
                            samples.append({
                                'mixture_audio': str(audio_file),
                                'target_video': str(video_file),
                                'speaker_id': item.name
                            })
        else:
            # Fallback to your avse_data structure
            scenes_path = self.avse_data_path / 'scenes'
            if scenes_path.exists():
                for scene_file in scenes_path.glob(f'{self.split}_*.json'):
                    with open(scene_file, 'r') as f:
                        scene_data = json.load(f)
                        for sample in scene_data:
                            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load mixture audio
        mixture_audio, sr = torchaudio.load(sample['mixture_audio'])
        if sr != 16000:
            mixture_audio = torchaudio.functional.resample(mixture_audio, sr, 16000)
        mixture_audio = mixture_audio.squeeze(0)  # Remove channel dim
        
        # Load target video
        target_video = self._load_video(sample['target_video'])
        
        # Load clean audio if available (for training)
        clean_audio = None
        if 'clean_audio' in sample:
            clean_audio, sr = torchaudio.load(sample['clean_audio'])
            if sr != 16000:
                clean_audio = torchaudio.functional.resample(clean_audio, sr, 16000)
            clean_audio = clean_audio.squeeze(0)
        
        return {
            'mixture_audio': mixture_audio,
            'target_video': target_video,
            'clean_audio': clean_audio,
            'speaker_id': sample.get('speaker_id', 'unknown')
        }
    
    def _load_video(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize to 224x224 for consistency
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            # Create dummy frames if video loading fails
            frames = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(25)]
        
        # Pad or trim to consistent length
        target_frames = min(len(frames), 50)  # Max 50 frames
        if len(frames) < target_frames:
            # Pad with last frame
            last_frame = frames[-1]
            frames.extend([last_frame] * (target_frames - len(frames)))
        else:
            frames = frames[:target_frames]
        
        # Convert to tensor
        frames = np.stack(frames)  # [T, H, W, C]
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        
        return frames

def collate_fn(batch):
    """Custom collate function for variable length sequences"""
    mixture_audios = []
    target_videos = []
    clean_audios = []
    speaker_ids = []
    
    max_audio_len = 0
    max_video_len = 0
    
    # Find max lengths
    for item in batch:
        max_audio_len = max(max_audio_len, item['mixture_audio'].size(0))
        max_video_len = max(max_video_len, item['target_video'].size(0))
    
    # Pad sequences
    for item in batch:
        # Pad audio
        mixture_audio = item['mixture_audio']
        if mixture_audio.size(0) < max_audio_len:
            padding = torch.zeros(max_audio_len - mixture_audio.size(0))
            mixture_audio = torch.cat([mixture_audio, padding])
        mixture_audios.append(mixture_audio)
        
        # Pad video
        target_video = item['target_video']
        if target_video.size(0) < max_video_len:
            last_frame = target_video[-1:].repeat(max_video_len - target_video.size(0), 1, 1, 1)
            target_video = torch.cat([target_video, last_frame])
        target_videos.append(target_video)
        
        # Handle clean audio
        if item['clean_audio'] is not None:
            clean_audio = item['clean_audio']
            if clean_audio.size(0) < max_audio_len:
                padding = torch.zeros(max_audio_len - clean_audio.size(0))
                clean_audio = torch.cat([clean_audio, padding])
            clean_audios.append(clean_audio)
        
        speaker_ids.append(item['speaker_id'])
    
    return {
        'mixture_audio': torch.stack(mixture_audios),
        'target_video': torch.stack(target_videos),
        'clean_audio': torch.stack(clean_audios) if clean_audios else None,
        'speaker_ids': speaker_ids
    }