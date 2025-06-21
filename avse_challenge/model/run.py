#!/usr/bin/env python3
"""
Unified runner script for Squeezeformer AVSE
Adapted for your existing COG-MHEAR repository structure
"""

import sys
import argparse
from pathlib import Path
import subprocess

def setup_environment():
    """Setup the environment with required packages"""
    requirements = [
        'torch>=1.9.0',
        'torchaudio',
        'opencv-python',
        'numpy',
        'pyyaml',
        'pesq',
        'pystoi'
    ]
    
    print("Installing required packages...")
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to install {req}")
    
    print("Environment setup complete!")

def train_model():
    """Run training"""
    from train import SqueezeformerTrainer
    
    config_path = Path(__file__).parent / 'config.yaml'
    trainer = SqueezeformerTrainer(config_path)
    trainer.train()

def evaluate_model(model_path, mode='dataset', **kwargs):
    """Run evaluation"""
    from eval import SqueezeformerEvaluator
    
    config_path = Path(__file__).parent / 'config.yaml'
    evaluator = SqueezeformerEvaluator(model_path, config_path)
    
    if mode == 'dataset':
        return evaluator.evaluate_on_dataset()
    else:
        evaluator.enhance_single_file(
            kwargs['mixture_audio'],
            kwargs['target_video'], 
            kwargs['output_path']
        )

def main():
    parser = argparse.ArgumentParser(description='Squeezeformer AVSE Runner')
    parser.add_argument('command', choices=['setup', 'train', 'eval', 'enhance'])
    
    # Training arguments
    parser.add_argument('--config', help='Config file path')
    
    # Evaluation arguments  
    parser.add_argument('--model_path', help='Trained model path')
    parser.add_argument('--mixture_audio', help='Input mixture audio')
    parser.add_argument('--target_video', help='Input target video')
    parser.add_argument('--output_path', help='Output enhanced audio path')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
    
    elif args.command == 'train':
        train_model()
    
    elif args.command == 'eval':
        if not args.model_path:
            print("Error: --model_path required for evaluation")
            return
        evaluate_model(args.model_path)
    
    elif args.command == 'enhance':
        if not all([args.model_path, args.mixture_audio, args.target_video, args.output_path]):
            print("Error: All arguments required for enhancement")
            return
        evaluate_model(
            args.model_path, 
            mode='single',
            mixture_audio=args.mixture_audio,
            target_video=args.target_video,
            output_path=args.output_path
        )

if __name__ == '__main__':
    main()