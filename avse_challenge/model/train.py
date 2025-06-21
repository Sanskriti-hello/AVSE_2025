import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import os
import sys
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


# Add parent directory to path for imports
#sys.path.append(str(Path(__file__).parent.parent))

from model.model import SqueezeformerAVSE
from model.dataset import MockCOGMHEARDataset as COGMHEARDataset, collate_fn
from baseline.evaluation.objective_evaluation import compute_metrics

USE_MOCK = True  # <-- Toggle this
#####
if USE_MOCK:
    from dataset import MockCOGMHEARDataset  # Add this to dataset.py
    self.train_dataset = MockCOGMHEARDataset(length=20)
    self.val_dataset = MockCOGMHEARDataset(length=5)
else:
    self.train_dataset = COGMHEARDataset(...)


class SqueezeformerTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = SqueezeformerAVSE().to(self.device)
        
        # Initialize datasets using your existing structure
        data_root = Path(self.config['data_root'])
        self.train_dataset = COGMHEARDataset(
            data_root, 
            split='train', 
            version=self.config['version']
        )
        self.val_dataset = COGMHEARDataset(
            data_root, 
            split='dev', 
            version=self.config['version']
        )
        
        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=1,  # Batch size 1 for validation
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Checkpoints directory
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            mixture_audio = batch['mixture_audio'].to(self.device)
            target_video = batch['target_video'].to(self.device)
            clean_audio = batch['clean_audio']
            
            if clean_audio is None:
                continue  # Skip if no clean reference
            
            clean_audio = clean_audio.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            enhanced_audio = self.model(mixture_audio, target_video)
            
            # Compute loss (align lengths)
            min_len = min(enhanced_audio.size(-1), clean_audio.size(-1))
            loss = self.criterion(
                enhanced_audio[..., :min_len], 
                clean_audio[..., :min_len]
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                mixture_audio = batch['mixture_audio'].to(self.device)
                target_video = batch['target_video'].to(self.device)
                clean_audio = batch['clean_audio']
                
                if clean_audio is None:
                    continue
                
                clean_audio = clean_audio.to(self.device)
                
                # Forward pass
                enhanced_audio = self.model(mixture_audio, target_video)
                
                # Compute loss
                min_len = min(enhanced_audio.size(-1), clean_audio.size(-1))
                loss = self.criterion(
                    enhanced_audio[..., :min_len], 
                    clean_audio[..., :min_len]
                )
                
                total_loss += loss.item()
                num_samples += 1
        
        return total_loss / num_samples if num_samples > 0 else float('inf')
    
    def train(self):
        print("Starting training...")
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss = self.validate()
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_dir / 'best_model.pth'
                )
                print("Saved best model!")
            
            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    },
                    self.checkpoint_dir / f'checkpoint_epoch_{epoch + 1}.pth'
                )

if __name__ == '__main__':
    config_path = Path(__file__).parent / 'config.yaml'
    trainer = SqueezeformerTrainer(config_path)
    trainer.train()