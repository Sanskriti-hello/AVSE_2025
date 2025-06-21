import torch
import torchaudio
from pathlib import Path
import sys
import yaml
import numpy as np

# Add paths for your existing evaluation
sys.path.append(str(Path(__file__).parent.parent))

from model.model import SqueezeformerAVSE
from model.dataset import MockCOGMHEARDataset as COGMHEARDataset, collate_fn
from baseline.evaluation.objective_evaluation import compute_metrics  # Use your existing metrics
from baseline.evaluation.objective_evaluation import evaluate_model  # Use your existing evaluator

class SqueezeformerEvaluator:
    def __init__(self, model_path, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = SqueezeformerAVSE().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"Loaded model from {model_path}")
    
    def evaluate_on_dataset(self):
        """Evaluate using your existing COG-MHEAR evaluation framework"""
        data_root = Path(self.config['data_root'])
        test_dataset = COGMHEARDataset(
            data_root, 
            split='test', 
            version=self.config['version']
        )
        
        results = []
        output_dir = Path('enhanced_outputs')
        output_dir.mkdir(exist_ok=True)
        
        print(f"Evaluating on {len(test_dataset)} samples...")
        
        with torch.no_grad():
            for i, sample in enumerate(test_dataset):
                mixture_audio = sample['mixture_audio'].unsqueeze(0).to(self.device)
                target_video = sample['target_video'].unsqueeze(0).to(self.device)
                
                # Enhance audio
                enhanced_audio = self.model(mixture_audio, target_video)
                enhanced_audio = enhanced_audio.squeeze(0).cpu()
                
                # Save enhanced audio
                output_path = output_dir / f'enhanced_{i:06d}.wav'
                torchaudio.save(output_path, enhanced_audio.unsqueeze(0), 16000)
                
                # Compute metrics if clean audio available
                if sample['clean_audio'] is not None:
                    clean_audio = sample['clean_audio']
                    
                    # Align lengths
                    min_len = min(enhanced_audio.size(0), clean_audio.size(0))
                    enhanced_aligned = enhanced_audio[:min_len].numpy()
                    clean_aligned = clean_audio[:min_len].numpy()
                    
                    # Use your existing metrics computation
                    metrics = compute_metrics(clean_aligned, enhanced_aligned, sample_rate=16000)
                    results.append(metrics)
                    
                    if i % 100 == 0:
                        print(f"Processed {i} samples, Current STOI: {metrics.get('stoi', 0):.3f}")
        
        # Compute average metrics
        if results:
            avg_metrics = {}
            for key in results[0].keys():
                avg_metrics[key] = np.mean([r[key] for r in results])
            
            print("\n=== Evaluation Results ===")
            for key, value in avg_metrics.items():
                print(f"{key.upper()}: {value:.4f}")
            
            return avg_metrics
        
        return {}
    
    def enhance_single_file(self, mixture_path, video_path, output_path):
        """Enhance a single audio-video pair"""
        # Load audio
        mixture_audio, sr = torchaudio.load(mixture_path)
        if sr != 16000:
            mixture_audio = torchaudio.functional.resample(mixture_audio, sr, 16000)
        mixture_audio = mixture_audio.squeeze(0).unsqueeze(0).to(self.device)
        
        # Load video (simplified)
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        
        if len(frames) == 0:
            print("Warning: No video frames loaded")
            return
        
        # Convert to tensor
        frames = frames[:50]  # Limit frames
        while len(frames) < 25:  # Minimum frames
            frames.append(frames[-1])
        
        target_video = torch.from_numpy(np.stack(frames))
        target_video = target_video.permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        
        # Enhance
        with torch.no_grad():
            enhanced_audio = self.model(mixture_audio, target_video)
        
        # Save
        enhanced_audio = enhanced_audio.squeeze(0).cpu()
        torchaudio.save(output_path, enhanced_audio.unsqueeze(0), 16000)
        print(f"Enhanced audio saved to {output_path}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config_path', default='config.yaml', help='Config file path')
    parser.add_argument('--mode', choices=['dataset', 'single'], default='dataset')
    parser.add_argument('--mixture_audio', help='Path to mixture audio (single mode)')
    parser.add_argument('--target_video', help='Path to target video (single mode)')
    parser.add_argument('--output_path', help='Output path (single mode)')
    
    args = parser.parse_args()
    
    config_path = Path(__file__).parent / args.config_path
    evaluator = SqueezeformerEvaluator(args.model_path, config_path)
    
    if args.mode == 'dataset':
        evaluator.evaluate_on_dataset()
    else:
        evaluator.enhance_single_file(
            args.mixture_audio, 
            args.target_video, 
            args.output_path
        )