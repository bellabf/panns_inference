import os
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import urllib.request
import ssl
import certifi

from .pytorch_utils import move_data_to_device
from .models import Cnn14, Cnn14_DecisionLevelMax
from .config import labels, classes_num

def create_folder(fd):
    """Create folder if it doesn't exist using pathlib."""
    Path(fd).mkdir(parents=True, exist_ok=True)

        
def get_filename(path):
    """Extract filename without extension using pathlib."""
    path = Path(path).resolve()
    return path.stem

def get_default_checkpoint_path(model_name):
    """Get default checkpoint path in a cross-platform way."""

    base_path = Path.home() / '.panns_data'
    return base_path / f'{model_name}.pth'

def download_checkpoint(checkpoint_path, zenodo_url):
    """Download checkpoint if it doesn't exist or is incomplete."""
    checkpoint_path = Path(checkpoint_path)
    create_folder(checkpoint_path.parent)

    if not checkpoint_path.exists() or checkpoint_path.stat().st_size < 3e8:
        print(f'Downloading checkpoint to {checkpoint_path}...')

    ssl_context = ssl.create_default_context(cafile=certifi.where())
        
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = int(downloaded * 100 / total_size)
        print(f"\rDownload progress: {percent}%", end='')
            
    try:
        urllib.request.urlretrieve(
                zenodo_url,
                str(checkpoint_path),
                reporthook=show_progress
        )
        print("\nDownload completed!")
    except Exception as e:
        print(f"\nError downloading checkpoint: {e}")
        if checkpoint_path.exists():
            checkpoint_path.unlink()  # Remove partial download
        raise


def load_checkpoint(checkpoint_path, device):
    """Safely load checkpoint with proper warning handling."""
    try:
        # Use weights_only=True for security
        return torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    except RuntimeError as e:
        # Fallback for older checkpoints that might not work with weights_only=True
        print("Warning: Unable to load with weights_only=True, falling back to default loading method")
        return torch.load(str(checkpoint_path), map_location=device)


class AudioTagging(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda'):
        """Audio tagging inference wrapper.
        """
        if checkpoint_path is None:
            checkpoint_path = get_default_checkpoint_path('Cnn14_mAP=0.431')
        
        checkpoint_path = Path(checkpoint_path)
        print('Checkpoint path:', checkpoint_path)
        
        # Download checkpoint if needed
        zenodo_url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
        download_checkpoint(checkpoint_path, zenodo_url)


        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num)
        else:
            self.model = model

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])

        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(audio, None)

        clipwise_output = output_dict['clipwise_output'].data.cpu().numpy()
        embedding = output_dict['embedding'].data.cpu().numpy()

        return clipwise_output, embedding


class SoundEventDetection(object):
    def __init__(self, model=None, checkpoint_path=None, device='cuda', interpolate_mode='nearest'):
        """Sound event detection inference wrapper.

        Args:
            model: None | nn.Module
            checkpoint_path: str
            device: str, 'cpu' | 'cuda'
            interpolate_mode, 'nearest' |'linear'
        """
        if checkpoint_path is None:
            checkpoint_path = get_default_checkpoint_path('Cnn14_DecisionLevelMax')
        
        checkpoint_path = Path(checkpoint_path)
        print('Checkpoint path:', checkpoint_path)

        zenodo_url = 'https://zenodo.org/record/3987831/files/Cnn14_DecisionLevelMax_mAP%3D0.385.pth?download=1'
        try:
            download_checkpoint(checkpoint_path, zenodo_url)
        except Exception as e:
            print(f"Failed to download checkpoint: {e}")
            raise


        if device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        
        self.labels = labels
        self.classes_num = classes_num

        # Model
        if model is None:
            self.model = Cnn14_DecisionLevelMax(sample_rate=32000, window_size=1024, 
                hop_size=320, mel_bins=64, fmin=50, fmax=14000, 
                classes_num=self.classes_num, interpolate_mode=interpolate_mode)
        else:
            self.model = model
        
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        self.model.load_state_dict(checkpoint['model'])


        # Parallel
        if 'cuda' in str(self.device):
            self.model.to(self.device)
            print('GPU number: {}'.format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)
        else:
            print('Using CPU.')

    def inference(self, audio):
        audio = move_data_to_device(audio, self.device)

        with torch.no_grad():
            self.model.eval()
            output_dict = self.model(
                input=audio, 
                mixup_lambda=None
            )

        framewise_output = output_dict['framewise_output'].data.cpu().numpy()

        return framewise_output
